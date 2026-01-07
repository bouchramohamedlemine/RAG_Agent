import logging 
from fastapi import FastAPI
import inngest 
import inngest.fast_api
from inngest.experimental import ai 
from dotenv import load_dotenv 
import uuid 
import os 
import datetime 
import uvicorn
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import *

 

# Load the env variables inside .env file
load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# When a file is uploaded through the front end, we send a request to the inngest server before our API.
# PDF -> Inngest server -> choose correct format -> our API
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag_app/ingest_pdf")
)



async def rag_ingest_pdf(ctx: inngest.Context):
    
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        """
         Load and chuck a pdf and return the result
        """
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # embed text
        vecs = embed_texts(chunks)

        # Generate a unique ID for all of these vectors
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]

        # Get a payload for every vector 
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        QdrantStorage().upsert(ids, vecs, payloads)

        return RAGUpsertResult(ingested=len(chunks))


    # Wrap chunking and embedding in steps
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)

    # Take pdantic model and convert it to python dictionary (serializable) and return it
    return ingested.model_dump()







# Create a function to search the vector DB
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag_app/query_pdf_ai")
)



async def query_pdf_ai(ctx: inngest.Context) -> RAGSearchResult:

    def _search(question: str, top_k: 5):
        # Make the question in the same format as the data by embeding it 
        query_vec = embed_texts([question])[0]

        store = QdrantStorage()
        found = store.search(query_vec, top_k)

        ctx.logger.info("--- DEBUGGING START ---")
        ctx.logger.info(f"Found Data: {found}")

        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])



    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # merge the contexts (i.e., the text chunks) into one block of text that we pass to ChatGPT
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    # The prompt 
    user_content = (
        "Use the following context to answer the question. \n\n"
        f"Context: \n{context_block}\n\n"
        f"Question: \n{question}\n\n"
        "Answer concisely using the context above"
    )
    
    # Use the adapter from inngest
    adapter = ai.openai.Adapter(
        auth_key = os.getenv("OPENAI_API_KEY"),
        model = "gpt-4o-mini"
    )

    response = await ctx.step.ai.infer(
        "llm-answer", 
        adapter = adapter, 
        body = {
                "max_tokens": 1024, 
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You answer questions using only the provided context."},
                    {"role": "user", "content": user_content}
                ]
            }
    )

    answer = response["choices"][0]["message"]["content"].strip()

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}




# Create our client 
app = FastAPI()

# This is where create functions
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, query_pdf_ai])


 

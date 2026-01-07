from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Break the PDF document into smaller peices and emmbed these pieces
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200) # chunk_overlap represents characters not words




def load_and_chunk_pdf(path):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)] # Only read text, not images etc.

    chunks = [] 
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks




def embed_texts(texts: list[str]) -> list[list[float]]:
    # Send request to openAI , pass all text chunked and embed (convert to vector to store in the vector DB)
    
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    # Pull out the embedding and ingore the other metadata 
    return [item.embedding for item in response.data]


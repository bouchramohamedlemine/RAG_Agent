# RAG Agent

A Retrieval-Augmented Generation (RAG) system that allows you to upload PDF documents and ask questions about their content. This app uses Qdrant for storage, Inngest for workflow orchestration, and OpenAI for both vector embeddings and LLM responses.

## Features

- **PDF Document Ingestion**: Upload PDFs and automatically chunk, embed, and store them in a vector database
- **Semantic Search**: Query your documents using natural language questions
- **AI-Powered Answers**: Get contextual answers from your documents using OpenAI's GPT models
- **Observability**: Monitor workflow steps and debug with Inngest's dashboard
- **Streamlit UI**: User-friendly web interface for uploading documents and asking questions


## Architecture

```
Streamlit UI → Inngest Events → FastAPI Backend → Qdrant Vector DB
                                      ↓
                                 OpenAI API
```

- **Streamlit**: Frontend interface for document uploads and queries
- **Inngest**: Workflow orchestration with step-by-step tracking and retry logic
- **FastAPI**: Backend API that processes PDFs and handles queries
- **Qdrant**: Vector database for storing and searching document embeddings
- **OpenAI**: Text embeddings (`text-embedding-3-large`) and LLM responses (`gpt-4o-mini`)


## Prerequisites

- Python 3.13+
- Node.js and npm (for Inngest CLI)
- Docker Desktop (for Qdrant)
- OpenAI API key

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bouchramohamedlemine/RAG_Agent.git
cd RAG_Agent
```

### 2. Set Up Python Environment

Create and activate a virtual environment:

```bash
python -m venv dev
source dev/bin/activate  # On Windows: dev\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
INNGEST_API_BASE=http://127.0.0.1:8288/v1  # Optional, defaults to this value
```

## Running the Application

The application requires **4 separate processes** to run simultaneously. Open 4 terminal windows/tabs.

### Terminal 1: Start Qdrant (Docker)

Start the Qdrant vector database using Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Note**: Make sure Docker Desktop is running before executing this command.

You can view the Qdrant dashboard at: http://127.0.0.1:6333/dashboard#/collections

### Terminal 2: Start FastAPI Backend

Start the FastAPI server with uvicorn:

```bash
uvicorn main:app --reload
```

This will start the server on **port 8000** (default). The `--reload` flag enables auto-reload during development.

### Terminal 3: Start Inngest Dev Server

Start the Inngest development server to manage workflow orchestration:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

**Important**: 
- This tells the Inngest CLI to connect to your FastAPI app on port 8000
- The `/api/inngest` endpoint is automatically created by `inngest.fast_api.serve()` in `main.py`
- You can access the Inngest dashboard at: http://127.0.0.1:8288/ to run functions, view steps, and debug

### Terminal 4: Start Streamlit Frontend

Start the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser at http://localhost:8501


## Usage

1. **Upload a PDF**: Use the Streamlit interface to upload a PDF document
2. **Wait for Processing**: The document will be automatically chunked, embedded, and stored in Qdrant
3. **Ask Questions**: Enter questions about the document content 
4. **View Results**: Get AI-powered answers with source citations

## How It Works

   - **Step 1**: Load and chunk the PDF into smaller text segments
   - **Step 2**: Generate embeddings using OpenAI's `text-embedding-3-large` model
   - **Step 3**: Store vectors in Qdrant with metadata (source, text chunks)
   - **Step 1**: Embed the user question using the same embedding model
   - **Step 2**: Search Qdrant for similar document chunks (semantic search)
   - **Step 3**: Retrieve top-k most relevant chunks
   - **Step 4**: Send context + question to OpenAI's `gpt-4o-mini` for answer generation
   - **Step 5**: Return answer with source citations


## Configuration

### Chunking Settings

Default chunking parameters (in `data_loader.py`):
- `chunk_size`: 1000 characters
- `chunk_overlap`: 200 characters


### Vector Database

- **Distance metric**: Cosine similarity
- **Embedding model**: text-embedding-3-large  
- **Vector dimension**: 3072

 

## Future Improvements

- **Multiple Document Support**: Support querying across multiple documents
- **Chunking Methods**: Experiment with semantic chunking, sliding windows, or hierarchical chunking
- **Hybrid Search**: Combine semantic search with keyword-based search
- **Multi-modal Support**: Extend to support images, tables, and other document types
- **Adaptive Top-K Retrieval**: The system should determine the optimal number of chunks to retrieve

import os
import sys
from pathlib import Path
from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PyMuPDFReader
import qdrant_client
import dotenv
dotenv.load_dotenv()

def setup_config():
    """Configure LlamaIndex with Google Gemini embedding model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=api_key,
    )
    
    # Use larger chunks to keep more context
    Settings.embed_model = embed_model
    Settings.chunk_size = 2048  # Much larger chunks
    Settings.chunk_overlap = 200
    
    Settings.embed_model = embed_model
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200

def load_resumes(resume_dir: str) -> List:
    """Load resume PDFs from the specified directory using PyMuPDF reader."""
    from pathlib import Path
    
    pdf_reader = PyMuPDFReader()
    documents = []
    
    pdf_files = list(Path(resume_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        try:
            docs = pdf_reader.load_data(str(pdf_path))
            documents.extend(docs)
            print(f"Loaded: {pdf_path.name}")
        except Exception as e:
            print(f"Error loading {pdf_path.name}: {e}")
    
    print(f"Loaded {len(documents)} resume documents")
    return documents

def setup_qdrant_vector_store(
    collection_name: str = "resumes",
    host: str = "localhost",
    port: int = 6333,
):
    """Initialize Qdrant vector store."""
    client = qdrant_client.QdrantClient(host=host, port=port)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    return vector_store, client

def ingest_resumes(resume_dir: str, collection_name: str = "resumes"):
    """Main function to ingest resumes into Qdrant."""
    print("Starting resume ingestion...")

    setup_config()
    documents = load_resumes(resume_dir)
    
    if not documents:
        print("No resume documents found!")
        return

    vector_store, client = setup_qdrant_vector_store(collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Creating index and ingesting {len(documents)} resumes into Qdrant...")
    
    # This process: Chunks -> Embeds -> Stores
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"✓ Successfully ingested {len(documents)} resumes into '{collection_name}'")
    return index, client, vector_store

if __name__ == "__main__":
    # This script lives under RAG/llamaindex/, but the PDFs live under RAG/resume-pdfs/
    resume_dir = Path(__file__).resolve().parents[1] / "resume-pdfs"

    if not resume_dir.exists():
        print(f"Error: Resume directory not found at {resume_dir}")
        sys.exit(1)

    ingest_resumes(str(resume_dir), collection_name="resumes")
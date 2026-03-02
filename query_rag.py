import os
import sys
from typing import List

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
)
# Corrected Imports
from llama_index.core.schema import NodeWithScore 
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI as Gemini
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

from dotenv import load_dotenv

load_dotenv()

def setup_config():
    """Configure LlamaIndex with Google Gemini embedding and Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        sys.exit(1)

    embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=api_key,
    )
    
    llm = Gemini(
        model="gemini-2.5-flash-lite",
        api_key=api_key,
        temperature=0.1,
    )
    
    # Global settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200

def connect_to_qdrant(collection_name: str = "resumes", host: str = "localhost", port: int = 6333):
    client = qdrant_client.QdrantClient(host=host, port=port)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return vector_store, client

def create_index(vector_store):
    # We pass the vector_store here so LlamaIndex knows where to look
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def query_resumes_with_details(query: str, top_k: int = 5, collection_name: str = "resumes"):
    print("=" * 80)
    print(f"Resume Search Query: {query}")
    print("=" * 80)
    
    setup_config()
    vector_store, _ = connect_to_qdrant(collection_name=collection_name)
    index = create_index(vector_store)
    
    # Retriever finds the raw text chunks (Top-K)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    
    print(f"\nFound {len(nodes)} matching resumes:\n")
    for i, node in enumerate(nodes, 1):
        print(f"--- Result {i} (Similarity: {node.score:.4f}) ---")
        # Metadata is preserved from the ingestion step
        print(f"Source: {node.metadata.get('file_name', 'Unknown')}")
        print(f"Content Preview:\n{node.text[:500]}...\n")
    
    return nodes

def interactive_query(collection_name: str = "resumes"):
    print("Resume Vector Database Query Tool")
    print("=" * 80)
    print("Type 'quit' or 'exit' to stop\n")
    
    setup_config()
    vector_store, _ = connect_to_qdrant(collection_name=collection_name)
    index = create_index(vector_store)
    
    # query_engine uses the LLM to synthesize an answer from retrieved nodes
    query_engine = index.as_query_engine(similarity_top_k=5)
    
    while True:
        user_query = input("\nEnter search query (e.g. 'Find a senior Python dev'): ").strip()
        if user_query.lower() in ["quit", "exit"]:
            break
        if not user_query: continue
        
        print("\n" + "-" * 40 + "\nLLM RESPONSE:\n" + "-" * 40)
        response = query_engine.query(user_query)
        print(f"{response}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
        query_resumes_with_details(search_query)
    else:
        interactive_query()
"""
Graph RAG Query Engine for existing Neo4j graph.
Combines graph subgraph retrieval + vector search, sending BOTH to LLM.
Custom implementation for Person/Skill/Thing schema.
"""

import os
import sys
from pathlib import Path
from neo4j import GraphDatabase

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI as Gemini
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import dotenv

dotenv.load_dotenv()


def setup_llm_and_embeddings():
    """Configure LlamaIndex with Gemini models."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
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
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 512
    
    return llm, embed_model


def setup_neo4j_connection():
    """Initialize direct Neo4j driver connection."""
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        print("Error: NEO4J_PASSWORD environment variable not set.")
        sys.exit(1)
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    
    # Test connection
    try:
        result = driver.execute_query("MATCH (n) RETURN count(n) as count LIMIT 1")
        node_count = result[0][0]['count']
        print(f"✓ Connected to Neo4j ({node_count} nodes)")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)
    
    return driver


def setup_vector_store():
    """Initialize Qdrant vector store connection."""
    try:
        client = QdrantClient(url="http://localhost:6333")
        vector_store = QdrantVectorStore(client=client, collection_name="resumes")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        print("✓ Connected to Qdrant vector store")
        return index
    except Exception as e:
        print(f"⚠ Could not connect to Qdrant: {e}")
        print("  Continuing with graph-only retrieval...")
        return None


def extract_entities_from_query(llm, query: str):
    """Use LLM to extract potential person names, skills, and concepts from query."""
    prompt = f"""Extract key entities from this query about a company's employee database.
Focus on:
- Person names (if mentioned)
- Skills (programming languages, technologies, frameworks)
- Job titles or roles
- Projects or accomplishments

Query: "{query}"

Return only a comma-separated list of the most important keywords/entities.
Examples: "Python, AWS, React" or "senior developer, cloud, kubernetes"

Keywords:"""
    
    response = llm.complete(prompt)
    keywords = [k.strip() for k in response.text.strip().split(',') if k.strip()]
    return keywords


def retrieve_graph_context(driver, keywords: list, max_depth: int = 2):
    """
    Retrieve subgraph context from Neo4j based on keywords.
    Searches for People with matching skills, titles, or names.
    """
    if not keywords:
        return []
    
    # Build flexible query that searches for skills first, then people
    cypher_query = """
    WITH $keywords AS keywords
    UNWIND keywords AS keyword
    
    // Find skills that match the keyword
    MATCH (s:Skill)
    WHERE toLower(s.name) CONTAINS toLower(keyword)
    
    // Find people who know those skills
    MATCH (p:Person)-[r:KNOWS]->(s)
    
    WITH DISTINCT p, collect(DISTINCT {skill: s.name, proficiency: r.proficiency, years: r.years_experience}) AS matched_skills
    
    // Get all skills for each person
    MATCH (p)-[r2:KNOWS]->(s2:Skill)
    WITH p, matched_skills, collect(DISTINCT {skill: s2.name, proficiency: r2.proficiency}) AS all_skills
    
    // Get sample accomplishments
    OPTIONAL MATCH (p)-[b:BUILT]->(t:Thing)
    WITH p, matched_skills, all_skills, collect(DISTINCT t.name)[0..3] AS accomplishments
    
    RETURN p.name AS name, 
           p.current_title AS title,
           p.department AS department,
           p.level AS level,
           p.years_experience AS years_experience,
           matched_skills,
           all_skills[0..10] AS all_skills,
           accomplishments
    ORDER BY p.name
    LIMIT 50
    """
    
    try:
        result = driver.execute_query(cypher_query, keywords=keywords)
        people_data = []
        
        for record in result[0]:
            person = {
                'name': record['name'],
                'title': record['title'],
                'department': record['department'],
                'level': record['level'],
                'years_experience': record['years_experience'],
                'matched_skills': [s['skill'] for s in record['matched_skills'] if s and s.get('skill')],
                'all_skills': [s['skill'] for s in record['all_skills'] if s and s.get('skill')],
                'accomplishments': record['accomplishments']
            }
            people_data.append(person)
        
        return people_data
    
    except Exception as e:
        print(f"Graph query error: {e}")
        return []


def retrieve_vector_context(vector_index, query: str, top_k: int = 5):
    """Retrieve similar resume chunks from vector store."""
    if not vector_index:
        return []
    
    try:
        retriever = vector_index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        context_chunks = []
        for node in nodes:
            context_chunks.append({
                'text': node.text[:500],  # Truncate for brevity
                'score': node.score,
                'source': node.metadata.get('file_name', 'Unknown')
            })
        
        return context_chunks
    
    except Exception as e:
        print(f"Vector search error: {e}")
        return []


def format_graph_context(people_data):
    """Format graph data into readable text."""
    if not people_data:
        return "No relevant graph data found."
    
    context = []
    context.append(f"Found {len(people_data)} relevant people in the knowledge graph:\n")
    
    for i, person in enumerate(people_data, 1):
        context.append(f"{i}. {person['name']} - {person['title']} ({person['department']}, {person['level']})")
        context.append(f"   Years experience: {person['years_experience']}")
        if person['matched_skills']:
            context.append(f"   Matched skills: {', '.join(person['matched_skills'])}")
        if person['all_skills']:
            other_skills = [s for s in person['all_skills'] if s not in person['matched_skills']]
            if other_skills:
                context.append(f"   Other skills: {', '.join(other_skills[:8])}")
        if person['accomplishments']:
            context.append(f"   Built: {', '.join(person['accomplishments'])}")
        context.append("")
    
    return "\n".join(context)


def format_vector_context(vector_chunks):
    """Format vector search results into readable text for LLM context (not for display)."""
    if not vector_chunks:
        return ""
    
    context = []
    context.append(f"\nAdditional relevant resume excerpts:")
    
    for chunk in vector_chunks:
        # Just include the text content, not the metadata/scores
        context.append(f"- {chunk['text'][:300]}")
    
    return "\n".join(context)


def query_graph(question: str):
    """
    Query using hybrid Graph RAG approach:
    1. Extract entities from question
    2. Retrieve subgraph from Neo4j
    3. Retrieve similar vectors from Qdrant
    4. Combine both contexts and send to LLM
    """
    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    
    # Setup
    llm, embed_model = setup_llm_and_embeddings()
    neo4j_driver = setup_neo4j_connection()
    vector_index = setup_vector_store()
    
    # Step 1: Extract entities
    print("\n[1/4] Extracting entities from question...")
    keywords = extract_entities_from_query(llm, question)
    print(f"  Keywords: {', '.join(keywords)}")
    
    # Step 2: Retrieve from graph
    print("\n[2/4] Retrieving subgraph from Neo4j...")
    graph_data = retrieve_graph_context(neo4j_driver, keywords)
    print(f"  Found {len(graph_data)} people")
    
    # Step 3: Retrieve from vector store
    print("\n[3/4] Retrieving from vector store...")
    vector_data = retrieve_vector_context(vector_index, question)
    print(f"  Found {len(vector_data)} relevant chunks")
    
    # Step 4: Combine contexts and generate answer
    print("\n[4/4] Generating answer from combined context...")
    
    graph_context = format_graph_context(graph_data)
    vector_context = format_vector_context(vector_data)
    
    combined_context = f"""KNOWLEDGE GRAPH DATA:
{graph_context}

{vector_context if vector_context else ''}"""
    
    # Generate final answer
    final_prompt = f"""You are a helpful HR assistant analyzing employee data.

User Question: {question}

PRIMARY DATA SOURCE - Knowledge Graph (use this for candidate names and structured info):
{graph_context}

SUPPLEMENTARY DATA - Resume Text Excerpts (use only to validate/enrich above candidates):
{vector_context if vector_context else 'No additional resume text available.'}

Instructions:
- Use ONLY the candidates listed in the Knowledge Graph section above
- Each candidate has a name, title, skills, and other structured data
- The resume text excerpts are for validation only - do not create new candidates from them
- If asking for top candidates, rank them based on how well their skills match the requirements
- Be specific with names, skills, and experience levels

Answer:"""
    
    response = llm.complete(final_prompt)
    
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(f"{response.text}\n")
    
    print("=" * 80)
    print(f"Context used: {len(graph_data)} people from graph, {len(vector_data)} vector chunks")
    print("=" * 80)
    
    neo4j_driver.close()
    
    return response.text


def interactive_mode():
    """Interactive query mode."""
    print("=" * 80)
    print("GRAPH RAG QUERY TOOL (Hybrid: Graph + Vector)")
    print("=" * 80)
    print("Queries the existing Neo4j graph built from JSON.")
    print("Combines graph subgraph + vector similarity for best results.")
    print("\nType 'quit', 'exit', or Ctrl+C to stop\n")
    
    while True:
        try:
            user_query = input("\nEnter query: ").strip()
            
            if user_query.lower() in ["quit", "exit"]:
                print("Exiting...")
                break
                
            if not user_query:
                print("Please enter a valid query.")
                continue
            
            query_graph(user_query)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Graph RAG query engine for existing Neo4j knowledge graph"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Single query to execute"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Interactive query mode (default if no --query)"
    )
    
    args = parser.parse_args()
    
    if args.query:
        query_graph(args.query)
    else:
        # Default to interactive mode
        interactive_mode()

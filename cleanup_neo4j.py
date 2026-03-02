"""
Clean (reset) the Neo4j database by deleting all nodes and relationships.
"""

import os
import sys
from neo4j import GraphDatabase
import dotenv

dotenv.load_dotenv()


def cleanup_neo4j():
    """Delete all nodes and relationships from Neo4j."""
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        print("Error: NEO4J_PASSWORD environment variable not set.")
        sys.exit(1)
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    
    # Test connection
    try:
        result = driver.execute_query("MATCH(n) RETURN count(n) as count")
        node_count = result[0][0]['count']
        print(f"Current nodes in graph: {node_count}")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)
    
    if node_count == 0:
        print("Graph is already empty.")
        driver.close()
        return
    
    # Confirm deletion
    confirm = input(f"\n⚠️  About to delete {node_count} nodes and all relationships. Continue? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("Cleanup cancelled.")
        driver.close()
        return
    
    # Delete all nodes and relationships
    try:
        print("\nDeleting all nodes and relationships...")
        driver.execute_query("MATCH (n) DETACH DELETE n")
        print("✓ Graph cleaned successfully!")
        
        # Verify
        result = driver.execute_query("MATCH(n) RETURN count(n) as count")
        final_count = result[0][0]['count']
        print(f"Nodes remaining: {final_count}")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    cleanup_neo4j()

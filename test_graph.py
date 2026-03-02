"""Quick test to verify Neo4j graph data."""
import os
from neo4j import GraphDatabase
import dotenv

dotenv.load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
)

# Test 1: Count Python developers
result = driver.execute_query(
    "MATCH (p:Person)-[:KNOWS]->(s:Skill {name: 'Python'}) RETURN count(p) as count, collect(p.name)[0..5] as sample_names"
)
print(f"Python developers: {result[0][0]['count']}")
print(f"Sample names: {result[0][0]['sample_names']}")

# Test 2: All skills
result = driver.execute_query(
    "MATCH (s:Skill) RETURN s.name ORDER BY s.name LIMIT 10"
)
print(f"\nSample skills: {[r[0] for r in result[0]]}")

driver.close()

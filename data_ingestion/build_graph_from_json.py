"""
Build Neo4j knowledge graph directly from extracted JSON data.
This approach guarantees data quality and entity deduplication.
Based on module2-graph-construction-and-retrieval.ipynb from neo4j-employee-graph. https://github.com/neo4j-product-examples/neo4j-employee-graph/blob/main/extract-resumes-to-people.py#L80
"""

import os
import sys
import json
from pathlib import Path
from neo4j import GraphDatabase, RoutingControl
import dotenv

dotenv.load_dotenv()


def chunks(xs, n=10):
    """Split list into chunks of size n."""
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def setup_neo4j_connection():
    """Connect to Neo4j database."""
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
        print(f"✓ Connected to Neo4j (existing nodes: {result[0][0]['count']})")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)
    
    return driver


def create_constraints(driver):
    """Create uniqueness constraints for nodes (Community Edition compatible)."""
    print("\nCreating constraints...")
    
    constraints = [
        ('Person', 'id'),
        ('Skill', 'name'),
        ('Thing', 'name'),
        ('Domain', 'name'),
        ('WorkType', 'name'),
    ]
    
    for node_type, property_name in constraints:
        # Use simple uniqueness constraint (works in Community Edition)
        constraint_name = f"{node_type.lower()}_{property_name}_unique"
        driver.execute_query(
            f'CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{node_type}) REQUIRE n.{property_name} IS UNIQUE',
            routing_=RoutingControl.WRITE
        )
        print(f"  ✓ {node_type}.{property_name}")


def load_people_nodes(driver, people_json):
    """Create Person nodes from JSON data."""
    print("\nCreating Person nodes...")
    total = 0
    
    for chunk in chunks(people_json):
        records = driver.execute_query(
            """
            UNWIND $records AS rec
            MERGE(person:Person {id:rec.id})
            SET person.name = rec.name,
                person.email = rec.email,
                person.current_title = rec.current_title,
                person.department = rec.department,
                person.level = rec.level,
                person.years_experience = rec.years_experience,
                person.location = rec.location
            RETURN count(rec) AS records_upserted
            """,
            routing_=RoutingControl.WRITE,
            result_transformer_=lambda r: r.data(),
            records=chunk
        )
        total += records[0]['records_upserted']
    
    print(f"✓ Created/updated {total} Person nodes")


def load_skills(driver, people_json):
    """Create Skill nodes and KNOWS relationships."""
    print("\nCreating Skills and relationships...")
    
    # Flatten skills from all people
    skills = []
    for person in people_json:
        tmp_skills = person['skills'].copy()
        for skill in tmp_skills:
            skill['personId'] = person['id']
        skills.extend(tmp_skills)
    
    print(f"  Found {len(skills)} skill entries")
    total = 0
    
    for chunk in chunks(skills):
        records = driver.execute_query(
            """
            UNWIND $records AS rec
            MATCH(person:Person {id:rec.personId})
            MERGE(skill:Skill {name:rec.skill.name})
            MERGE(person)-[r:KNOWS]->(skill)
            SET r.proficiency = rec.proficiency,
                r.years_experience = rec.years_experience,
                r.context  = rec.context,
                r.is_primary = rec.is_primary
            RETURN count(rec) AS records_upserted
            """,
            routing_=RoutingControl.WRITE,
            result_transformer_=lambda r: r.data(),
            records=chunk
        )
        total += records[0]['records_upserted']
    
    print(f"✓ Created {total} KNOWS relationships")


def load_accomplishments(driver, people_json):
    """Create Thing/Domain/WorkType nodes and accomplishment relationships."""
    print("\nCreating Accomplishments...")
    
    # Flatten accomplishments from all people
    accomplishments = []
    for person in people_json:
        tmp_accomplishments = person['accomplishments'].copy()
        for accomplishment in tmp_accomplishments:
            accomplishment['personId'] = person['id']
        accomplishments.extend(tmp_accomplishments)
    
    print(f"  Found {len(accomplishments)} accomplishment entries")
    total = 0
    
    for chunk in chunks(accomplishments):
        records = driver.execute_query(
            """
            UNWIND $records AS rec

            //match people
            MATCH(person:Person {id:rec.personId})

            //merge accomplishments
            MERGE(thing:Thing {name:rec.thing.name})
            MERGE(person)-[r:BUILT]->(thing)
            SET r.impact_description = rec.impact_description,
                r.year = rec.year,
                r.role  = rec.role,
                r.duration = rec.duration,
                r.team_size = rec.team_size,
                r.context  = rec.context

            //merge domain and work type
            MERGE(domain:Domain {name:rec.thing.domain})
            MERGE(thing)-[:IN]->(domain)
            MERGE(workType:WorkType {name:rec.thing.type})
            MERGE(thing)-[:OF]->(workType)

            RETURN count(rec) AS records_upserted
            """,
            routing_=RoutingControl.WRITE,
            result_transformer_=lambda r: r.data(),
            records=chunk
        )
        total += records[0]['records_upserted']
    
    print(f"✓ Created {total} accomplishment relationships")


def verify_graph(driver):
    """Print graph statistics."""
    print("\n" + "="*80)
    print("GRAPH STATISTICS")
    print("="*80)
    
    # Node counts
    result = driver.execute_query(
        """
        MATCH (n)
        RETURN labels(n)[0] as label, count(*) as count
        ORDER BY count DESC
        """
    )
    
    print("\nNodes by type:")
    for record in result[0]:
        print(f"  {record['label']}: {record['count']}")
    
    # Relationship counts
    result = driver.execute_query(
        """
        MATCH ()-[r]->()
        RETURN type(r) as relationship, count(*) as count
        ORDER BY count DESC
        """
    )
    
    print("\nRelationships by type:")
    for record in result[0]:
        print(f"  {record['relationship']}: {record['count']}")
    
    # Python developers
    result = driver.execute_query(
        """
        MATCH (p:Person)-[:KNOWS]->(s:Skill {name: 'Python'})
        RETURN count(p) as python_devs
        """
    )
    
    python_count = result[0][0]['python_devs']
    print(f"\n✓ Python developers found: {python_count}")
    print("="*80)


def main():
    """Build the knowledge graph from extracted JSON."""
    print("="*80)
    print("BUILDING NEO4J KNOWLEDGE GRAPH FROM JSON")
    print("="*80)
    
    # Load JSON data (JSON lives at RAG/extracted-people-data.json)
    rag_root = Path(__file__).resolve().parents[1]
    json_file = rag_root / "extracted-people-data.json"
    
    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        print("If you don't have it yet, generate/copy it (see neo4j-employee-graph/) and place it at RAG/extracted-people-data.json.")
        sys.exit(1)
    
    print(f"\nLoading data from {json_file.name}...")
    with open(json_file, 'r') as f:
        people_json = json.load(f)
    
    print(f"✓ Loaded {len(people_json)} people records")
    
    # Connect to Neo4j
    driver = setup_neo4j_connection()
    
    try:
        # Build graph
        create_constraints(driver)
        load_people_nodes(driver, people_json)
        load_skills(driver, people_json)
        load_accomplishments(driver, people_json)
        
        # Verify
        verify_graph(driver)
        
        print("\n✓ Graph construction complete!")
        print("\nNext steps:")
        print("  1. Open Neo4j Browser: http://localhost:7474")
        print("  2. Run Cypher queries:")
        print("     MATCH (p:Person)-[:KNOWS]->(s:Skill {name: 'Python'}) RETURN p.name")
        print("  3. Or use Python to query:")
        print("     python query_graph_direct.py")
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()

# RAG Demo (Vector RAG + Graph RAG)

This folder contains a practical, demo-friendly RAG playground built around **resume PDFs**. The goal is to answer HR-style questions like:

- “How many Python developers do we have?”
- “Who has AWS + React experience?”
- “Who is most similar to X?”

We ended up with two complementary retrieval paths:

1. **Vector RAG (Qdrant)** for semantic search over resume text.
2. **Graph RAG (Neo4j)** for precise counting, relationships, and structured queries.

---

## What We’re Trying To Achieve

### Vector RAG

- Ingest all PDFs under `resume-pdfs/`.
- Extract _real text_ from PDFs (not file metadata).
- Chunk + embed text and store in Qdrant.
- Query using Gemini LLM for summarization/answers.

### Graph RAG

- Build a Neo4j graph representing people, skills, accomplishments, etc.
- Query the graph for exact questions (counts, skill intersections, relationships).
- Optionally combine graph results with vector search for “fuzzy” queries.

---

## Prerequisites (What You Need Installed)

- Python 3.11+ (3.12 recommended)
- Docker Desktop (for Neo4j and Qdrant)
- A Google Gemini API key (`GOOGLE_API_KEY`)

Optional:

- Neo4j Browser (comes with the Neo4j container UI)

---

## Setup (macOS / Linux / Windows)

### 1) Create a virtual environment and install dependencies

From the `RAG/` folder:

macOS / Linux:

- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

Windows (PowerShell):

- `py -m venv venv`
- `./venv/Scripts/Activate.ps1`
- `pip install -r requirements.txt`

### 2) Configure environment variables

Create a `.env` file in `RAG/` with:

- `GOOGLE_API_KEY=...`
- `NEO4J_URI=bolt://localhost:7687`
- `NEO4J_USERNAME=neo4j`
- `NEO4J_PASSWORD=...`

### 3) Start Qdrant (Vector DB)

If you don’t already have Qdrant running on `http://localhost:6333`, start it with Docker:

- `docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest`

### 4) Start Neo4j (Graph DB)

If you don’t already have Neo4j running, start it with Docker:

- `docker run -d --name neo4j-graph -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<your_password_here> neo4j:5.15.0`

Then open Neo4j Browser:

- `http://localhost:7474`

---

---

## How To Run (Suggested Demo Flow)

### 1) Vector RAG

- Ingest (build Qdrant collection): `python ./data_ingestion/ingest_qadrant.py`
- Query (interactive): `python ./query_vector.py`
- Query (single shot): `python ./query_vector.py "Find a senior Python dev"`

### 2) Graph (from JSON)

- Build graph from JSON: `python ./data_ingestion/build_graph_from_json.py`
- Sanity check Neo4j contents: `python ./test_graph.py`

Notes:

- The graph builder expects `extracted-people-data.json` in the project root.
- If that file doesn't exist yet, generate/copy it from the `neo4j-employee-graph/` demo.
- **To reset the graph**, wipe it first then rebuild:
  - `python ./cleanup_neo4j.py`
  - `python ./data_ingestion/build_graph_from_json.py`
- This deterministic JSON → graph approach is far more reliable than building directly from PDFs.

### 3) Hybrid Query (Graph + Vector)

**Recommended: True Graph RAG (combines both graph + vector)**

- Run interactive: `python ./query_existing_graph.py`
- Or single query: `python ./query_existing_graph.py --query "How many Python developers do I have?"`

This uses a hybrid retrieval pattern where both graph subgraph results and vector similarity results are sent to the LLM together for synthesis.

**Alternative: LLM-routed query (chooses graph OR vector)**

- Run interactive: `python ./query_graph_rag.py`
- Or single query: `python ./query_graph_rag.py --query "How many Python developers do I have?"`

This implementation uses an LLM to decide between graph or vector search (not both).

## Environment

Expected `.env` values:

- `GOOGLE_API_KEY`
- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME` (default: `neo4j`)
- `NEO4J_PASSWORD`

Qdrant expected at:

- `http://localhost:6333`

Neo4j expected at:

- Browser: `http://localhost:7474`
- Bolt: `bolt://localhost:7687`

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

### Optional: LlamaIndex Graph RAG (build graph from PDFs)

This approach builds a knowledge graph directly from resume PDFs via LlamaIndex. It is **not recommended** for demo use — entity extraction is unpredictable, produces duplicate/inconsistent nodes, and is hard to validate.

The `llamaindex/` folder and its scripts have been removed in favour of the deterministic JSON → graph approach in `data_ingestion/`.

---

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

## Trials & Errors (What We Learned)

### 1) PDF ingestion “worked”… but returned garbage

**Symptom:** Vector search answers were nonsensical (“no Python developers”), and retrieved chunks contained PDF structure/metadata instead of resume text.

**Root cause:** LlamaIndex’s default directory reader didn’t reliably extract text from PDFs in our setup,and my lack of knowledge in llamaindex

**Fix:** Switched to a dedicated PDF text reader (`PyMuPDFReader`), which produced real textual content for embeddings.

---

### 2) Embeddings model choice impacted both quality and cost

We experimented with:

- Local embeddings (free) vs Gemini embeddings (paid)
- Gemini LLM for generation

**Trade-off:** Gemini embeddings can improve quality but are billable. Local embeddings avoid embedding costs.

---

### 3) LlamaIndex “Graph RAG directly from PDFs” was not dependable here

**Symptom:** Asking “How many Python developers?” via the LlamaIndex knowledge graph path returned drastically incorrect counts.

**Why it happens:** Automatic entity/relationship extraction from raw text can be inconsistent (duplicates, missed entities, schema drift). It’s hard to validate/repair before data hits the graph.

**Decision:** Don’t build the graph from raw PDFs for the demo. Instead, extract structured entities first.

---

### 4) The winning pattern: JSON first → Graph second

We saw a much more reliable pattern in `neo4j-employee-graph/`:

1. Extract structured entities/relations into JSON (`extracted-people-data.json`).
2. Create graph nodes and relationships deterministically using Cypher.

**Benefit:** Stable schema + accurate counts (e.g., Python developers = 28), and predictable queries.

---

### 6) “Text-to-Cypher” needs guardrails

When using an LLM to generate Cypher, you need:

- schema context
- strict output format (JSON only)
- robust parsing (handle code fences)
- fallback strategy (vector search) if graph query fails

---

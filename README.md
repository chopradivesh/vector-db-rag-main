# VectorDB in Python — a vector database and RAG pipeline built from scratch

A Python rewrite of a C++ educational project. It implements **three nearest-neighbour
search algorithms by hand** (Brute Force, KD-Tree, HNSW) and **three distance metrics**,
and adds a **RAG (Retrieval-Augmented Generation)** pipeline that uses a local LLM through Ollama.
It also ships the original web UI.

---

## 1. Project overview

A *vector database* stores lists of numbers (vectors / embeddings) and answers one question fast:
**"which stored vectors are closest to this query vector?"** When the vectors come from an
embedding model, "close" means "similar in meaning". Semantic search and RAG are built on this.

This project has two databases:

| Database | Vectors | Where they come from | Search |
|---|---|---|---|
| `VectorDB` (demo) | 20 × 16-D | **hand-written** (4 topics × 4 dims) | Brute force, KD-Tree **or** HNSW, any metric |
| `DocumentDB` | N × 768-D | **real embeddings** from Ollama `nomic-embed-text` | Brute force (< 10 chunks) or HNSW, cosine |

## 2. Features

- Brute-force, KD-Tree and HNSW k-NN search, all written from scratch
- Cosine, Euclidean and Manhattan distance
- Benchmark endpoint that times all three algorithms on the same query
- HNSW graph statistics (nodes/edges per layer)
- Word-based text chunking (250 words, 30-word overlap)
- RAG: embed → retrieve → prompt → generate, using Ollama running locally
- REST API with the same routes as the C++ server, plus the original web UI (PCA scatter plot, chat)
- 61 automated tests

## 3. Architecture

```
vectordb-python/
├── main.py                      starts the web server
├── src/
│   ├── config.py                settings from environment / .env
│   ├── indexing/                THE ALGORITHMS
│   │   ├── distances.py         cosine, euclidean, manhattan
│   │   ├── brute_force.py       exact O(N·d) scan
│   │   ├── kd_tree.py           exact, space-partitioning tree
│   │   └── hnsw.py              approximate, multi-layer graph
│   ├── vector_db/               STORAGE
│   │   ├── models.py            VectorItem, DocItem
│   │   ├── vector_db.py         demo DB: one item in all 3 indexes
│   │   ├── document_db.py       document-chunk DB (HNSW, cosine)
│   │   └── demo_data.py         the 20 demo vectors
│   ├── embeddings/
│   │   └── keyword_embedder.py  text → 16-D demo vector (keyword rules, NOT ML)
│   ├── llm/
│   │   └── ollama_client.py     HTTP client: embed() and generate()
│   ├── rag/
│   │   ├── chunker.py           split text into overlapping chunks
│   │   └── pipeline.py          ingest / retrieve / ask
│   └── api/
│       └── server.py            FastAPI routes (same JSON as the C++ server)
├── static/index.html            original web UI (unchanged)
├── examples/                    runnable scripts
└── tests/                       pytest suite
```

Dependencies only flow downwards: `api → rag → vector_db → indexing`.
The indexes know nothing about HTTP, and the RAG pipeline knows nothing about FastAPI.

## 4. Installation

Requires **Python 3.11+**.

```bash
cd vectordb-python
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The RAG part also needs [Ollama](https://ollama.com):

```bash
ollama pull nomic-embed-text       # embedding model (~274 MB)
ollama pull llama3.2               # language model (~2 GB)
ollama serve                       # if it's not already running
```

The search algorithms, demo, tests and examples all work **without** Ollama.

## 5. Environment variables

Copy `.env.example` to `.env` if you want to change anything. No API keys are needed.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `GEN_MODEL` | `llama3.2` | answer model (try `llama3.2:1b` on slow laptops) |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | web server (the UI expects port 8080) |

## 6. How to run

```bash
python main.py                      # web UI + API at http://localhost:8080
python -m pytest                    # run tests
python -m examples.demo_search      # search demo vectors (no Ollama)
python -m examples.benchmark        # speed/recall of the 3 algorithms at larger N and d
python -m examples.rag_demo         # full RAG with Ollama
```

### Live demo (Render)

The project deploys to [Render](https://render.com) using `render.yaml`
(New → Blueprint → pick this repo). The live demo supports **vector search only**:
search, benchmark and the HNSW graph view. Document upload and question answering
need Ollama, which runs locally, so they show "Ollama offline" online.
The free plan sleeps when idle (the first request can take about a minute), and
vectors added online are lost on restart.

## 7. Example usage (Python)

```python
from src.vector_db.vector_db import VectorDB
from src.vector_db.demo_data import load_demo
from src.embeddings.keyword_embedder import text_to_demo_embedding

db = VectorDB(16)
load_demo(db)
out = db.search(text_to_demo_embedding("sushi", jitter=0), k=3, metric="cosine", algo="hnsw")
for distance, item in out.hits:
    print(round(distance, 4), item.metadata)
```

REST:

```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&algo=hnsw"
curl -X POST localhost:8080/doc/insert -H 'Content-Type: application/json' -d '{"title":"OS notes","text":"..."}'
curl -X POST localhost:8080/doc/ask    -H 'Content-Type: application/json' -d '{"question":"What is a deadlock?","k":3}'
```

| Method | Route | Purpose |
|---|---|---|
| GET | `/search?v=..&k=5&metric=cosine&algo=hnsw` | k-NN on demo vectors |
| POST | `/insert` `{metadata, category, embedding}` | add a demo vector |
| DELETE | `/delete/{id}` | remove a demo vector |
| GET | `/items`, `/stats`, `/hnsw-info` | list / stats / graph structure |
| GET | `/benchmark?v=..&k=5&metric=cosine` | time all 3 algorithms |
| POST | `/doc/insert` `{title, text}` | chunk + embed + store |
| GET | `/doc/list` · DELETE `/doc/delete/{id}` | manage chunks |
| POST | `/doc/search` `{question, k}` | retrieval only |
| POST | `/doc/ask` `{question, k}` | full RAG answer |
| GET | `/status` | Ollama status, counts |

## 8. The algorithms

**Distance metrics** (`indexing/distances.py`), all O(d):
- *Euclidean*: `sqrt(Σ(aᵢ−bᵢ)²)`, straight-line distance.
- *Manhattan*: `Σ|aᵢ−bᵢ|`, grid distance.
- *Cosine distance*: `1 − (a·b)/(|a||b|)`. It measures only the angle, not the length, which makes it the standard metric for text embeddings. A zero vector returns 1.0.

**Brute force** compares the query with every vector, sorts, and keeps the top k. It is always exact, so it serves as the ground truth.

**KD-Tree** is a binary tree. The node at depth *t* splits on dimension `t % d`. The search goes down the query's side first. It only visits the other side when `|query[axis] − node[axis]|` (the shortest possible distance across the split) is smaller than the current k-th best, because otherwise that whole subtree can be skipped. This works very well in low dimensions. In high dimensions the check almost never skips anything, so the search ends up close to brute force.

**HNSW** is a layered graph, similar to a skip list. Every node is on layer 0. A random, exponentially shrinking share of nodes also appears on higher layers, which act as "highways".
- *Search:* start at the top-layer entry point, walk greedily towards the query, and drop down a layer each time you can't get closer. On layer 0, run a beam search that keeps the `ef` best candidates, using two heaps.
- *Insert:* pick a random level, search down to it in the same way, then connect the node to its `M` closest candidates on each layer (`2M` on layer 0) in both directions. If a neighbour ends up with too many links, it keeps only its closest ones.
- The result is approximate: it can miss a true neighbour in exchange for speed.

## 9. RAG pipeline

```
INGEST   text ─► chunk_text (250 words, 30 overlap) ─► Ollama embed (768-D) ─► DocumentDB (HNSW, cosine)
ASK      question ─► Ollama embed ─► DocumentDB.search(k, distance ≤ 0.7) ─► prompt(context + question) ─► llama3.2 ─► answer
```

The question must be embedded with the **same model** as the chunks, or the distances mean nothing.
The prompt (unchanged from the original) lets the model **fall back on its own knowledge**, so answers
are *not* guaranteed to come only from your documents.

## 10. Example input / output

`python -m examples.demo_search` (real output):

```
Query: "binary tree"
  bruteforce    19 us  Binary Search Tree (0.0021), Linked List (0.0037), Hash Table (0.0038)
  kdtree        26 us  Binary Search Tree (0.0021), Linked List (0.0037), Hash Table (0.0038)
  hnsw          29 us  Binary Search Tree (0.0021), Linked List (0.0037), Hash Table (0.0038)
```

`python -m examples.benchmark` (real output; times are per query in pure Python):

```
N=5000  d=4    brute=  2022us  kdtree=  138us  hnsw=  168us  hnsw recall@10=1.00
N=5000  d=64   brute= 10125us  kdtree=10539us  hnsw= 2418us  hnsw recall@10=0.92
N=5000  d=256  brute= 37931us  kdtree=38607us  hnsw= 9325us  hnsw recall@10=0.77
```

In low dimensions the KD-Tree wins. At 64 and 256 dimensions it is no faster than brute force,
while HNSW stays about 4× faster but starts to miss some neighbours (recall < 1). Raising `ef` trades speed back for recall.

## 11. Time and space complexity

| | Build / insert | Query | Exact? | Memory |
|---|---|---|---|---|
| Brute force | O(1) | O(N·d + N log N) | yes | O(N·d) |
| KD-Tree | O(log N) avg, O(N) worst | ~O(log N) low-d → O(N·d) high-d | yes | O(N·d) |
| HNSW | ~O(log N · ef_c · M · d) | ~O(log N · ef · M · d) | no (approx.) | O(N·d + N·M) |
| Chunking | — | O(W) words | — | O(W) |

Deleting: brute force O(N); KD-Tree rebuilds the whole tree, O(N log N); HNSW scans all edges, O(N·M).

## 12. Known limitations & future improvements

- **In-memory only.** Everything is lost on restart. Next step: save to disk (e.g. JSON or pickle).
- HNSW delete removes edges but doesn't reconnect the neighbours left behind, and never lowers `top_layer`.
- HNSW neighbour selection is the paper's "simple" version (closest M), not its diversity heuristic.
- The demo HNSW graph is always **built** with cosine even when searched with another metric.
- Pure-Python distances are slow; numpy vectorisation would make them 10–100× faster.
- The KD-Tree isn't balanced (no median split), so the insert order decides its shape.
- The answer isn't streamed; the UI simulates a typewriter effect.
- Could add metadata filtering, persistence, and a `/api/embed` batch endpoint.

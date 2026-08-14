# Delhi Metro RAG Assistant

Extends the [Delhi Metro Graph Algorithm Visualizer](../delhi-metro-graph-project)
with a RAG chat interface. Factual questions ("Is Kashmere Gate an
interchange?") are answered by retrieving over auto-generated station
docs. Route questions ("shortest path from Rithala to Rajouri Garden")
are routed to the real Dijkstra implementation instead of letting the
LLM guess — so distances/paths are always exactly correct, never
hallucinated.

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)
(sign up, no card needed, generous free tier).

## Run

```bash
streamlit run app.py
```

Paste your Groq API key into the sidebar, then ask things like:
- "Is Rajiv Chowk an interchange station?"
- "What lines does Kashmere Gate connect?"
- "Shortest path from Rithala to Rajouri Garden"

## How it works

1. **Doc generation** (`src/rag_pipeline.py`) — one text doc per station,
   built from the graph you already constructed (line, interchange
   status, direct neighbors + distances). No scraping needed.
2. **Retrieval** — `sentence-transformers` embeds all docs once at
   startup; queries are matched via cosine similarity (plain numpy —
   at ~260 docs, no need for FAISS).
3. **Routing** (`src/router.py`) — detects "X to Y" style questions,
   fuzzy-matches both station names against the real station list, and
   if matched, runs your existing `dijkstra()` instead of retrieval.
4. **Generation** (`src/llm.py`) — Groq's API turns the retrieved
   context (or computed route) into a natural-language answer, grounded
   in real data.

## Project structure

```
metro-rag/
├── app.py                  # Streamlit UI
├── data/
│   ├── Delhi metro.csv
│   └── data_loader.py       # reused as-is from the original project
├── src/
│   ├── graph_algorithms.py  # reused as-is (Graph, dijkstra, etc.)
│   ├── rag_pipeline.py      # NEW: doc generation + retrieval
│   ├── router.py            # NEW: route-query detection
│   └── llm.py                # NEW: Groq API call
└── requirements.txt
```

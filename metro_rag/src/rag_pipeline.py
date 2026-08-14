"""
RAG retrieval over the Delhi Metro graph.

Instead of scraping external text, we GENERATE one document per station
from data we already have (line, interchange status, neighbors, distances).
This is cheap, accurate (no hallucination risk in the source docs), and
reuses the Graph class you already built.

Retrieval is plain numpy cosine similarity, not FAISS -- with ~260
stations, a brute-force search over all embeddings is microseconds, and
it's easier to see exactly what's happening (no index internals to trust).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.graph_algorithms import Graph


def build_station_docs(graph: Graph, records) -> list[dict]:
    """
    One text doc per canonical station: which line(s) it's on, whether it's
    an interchange, and its direct neighbors with distances.

    `records` is the raw list from data_loader.load_records() -- we use it
    to recover which lines each canonical station sits on (a station's
    line membership isn't stored on the Graph object itself).
    """
    lines_by_station: dict[str, set[str]] = {}
    for r in records:
        lines_by_station.setdefault(r["canonical_name"], set()).add(r["line"])

    docs = []
    for station in sorted(graph.stations):
        lines = sorted(lines_by_station.get(station, []))
        neighbors = graph.neighbors(station)
        neighbor_text = "; ".join(
            f"{n} ({w:.2f} km)" for n, w in sorted(neighbors, key=lambda x: x[1])
        )
        is_interchange = len(lines) > 1

        text = (
            f"{station} is a station on the Delhi Metro. "
            f"It is served by: {', '.join(lines)}. "
            f"{'This is an interchange station between lines. ' if is_interchange else ''}"
            f"Directly connected stations: {neighbor_text if neighbor_text else 'none recorded'}."
        )
        docs.append({"station": station, "text": text})
    return docs


class StationRetriever:
    """Embeds station docs once, then answers nearest-neighbor queries via cosine similarity."""

    def __init__(self, docs: list[dict], model_name: str = "all-MiniLM-L6-v2"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        texts = [d["text"] for d in docs]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.embeddings = np.asarray(embeddings)  # (num_docs, dim), unit-normalized

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        # cosine similarity == dot product, since both sides are unit-normalized
        scores = self.embeddings @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {**self.docs[i], "score": float(scores[i])}
            for i in top_idx
        ]

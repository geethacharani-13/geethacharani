"""
Decides whether a question should be answered by:
  (a) computing a real shortest path with your existing Dijkstra function, or
  (b) retrieving station docs (plain RAG).

Why this matters: an LLM asked "shortest path from Rithala to Rajouri
Garden" will happily produce a plausible-looking but WRONG path, because
it has no access to the actual graph. Detecting this case and routing to
real Dijkstra removes that failure mode entirely.
"""

import difflib
import re

from src.graph_algorithms import Graph, dijkstra

_PATH_TRIGGER_WORDS = ("shortest", "fastest", "path", "route", "distance", "how far", "how do i get")


def _fuzzy_match_station(name: str, known_stations: list[str], cutoff: float = 0.72) -> str | None:
    matches = difflib.get_close_matches(name.strip(), known_stations, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def extract_route_query(question: str, known_stations: list[str]) -> tuple[str, str] | None:
    """
    Looks for a "from X to Y" / "X to Y" pattern and fuzzy-matches both
    sides against real station names. Returns (start, end) or None.
    """
    q_lower = question.lower()
    if not any(word in q_lower for word in _PATH_TRIGGER_WORDS):
        return None

    match = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if not match:
        match = re.search(r"(.+?)\s+to\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if not match:
        return None

    raw_start, raw_end = match.group(1).strip(), match.group(2).strip()
    start = _fuzzy_match_station(raw_start, known_stations)
    end = _fuzzy_match_station(raw_end, known_stations)

    if start and end:
        return start, end
    return None


def answer_route_query(graph: Graph, start: str, end: str) -> str:
    """Runs real Dijkstra and formats the result as context for the LLM to phrase."""
    path, dist, _ = dijkstra(graph, start, end)
    if path is None:
        return f"No path found between {start} and {end} (they may be in disconnected parts of the network)."
    stops = len(path) - 1
    route_str = " -> ".join(path)
    return (
        f"Computed shortest path (Dijkstra, real track distance) from {start} to {end}: "
        f"{route_str}. Total distance: {dist:.2f} km across {stops} stops."
    )

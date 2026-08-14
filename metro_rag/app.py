"""
Delhi Metro RAG Assistant.

Ask questions about the network. Factual questions ("Which lines does
Rajouri Garden connect?") are answered via retrieval over generated
station docs. Route questions ("shortest path from Rithala to Rajouri
Garden") are answered by running the real Dijkstra implementation --
not the LLM guessing -- so the numbers are always correct.

Run with: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from data.data_loader import load_records
from src.graph_algorithms import Graph
from src.rag_pipeline import build_station_docs, StationRetriever
from src.router import extract_route_query, answer_route_query
from src.llm import generate_answer


st.set_page_config(page_title="Delhi Metro RAG Assistant", page_icon="🚇")
st.title("🚇 Delhi Metro RAG Assistant")
st.caption(
    "Ask about stations, lines, interchanges, or routes. "
    "Route questions run real Dijkstra; everything else is retrieval-augmented."
)


@st.cache_resource(show_spinner="Loading network and building embeddings...")
def load_pipeline():
    records = load_records(os.path.join(os.path.dirname(__file__), "data", "Delhi metro.csv"))
    graph = Graph().build_from_csv()
    docs = build_station_docs(graph, records)
    retriever = StationRetriever(docs)
    return graph, retriever


graph, retriever = load_pipeline()

with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com")
    top_k = st.slider("Docs retrieved per question", 1, 5, 3)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("e.g. 'shortest path from Rithala to Rajouri Garden' or 'is Kashmere Gate an interchange?'")

if question:
    if not api_key:
        st.error("Add your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            route = extract_route_query(question, list(graph.stations))

            if route:
                start, end = route
                context = answer_route_query(graph, start, end)
                st.caption(f"🧭 Routed to Dijkstra: {start} → {end}")
            else:
                retrieved = retriever.retrieve(question, top_k=top_k)
                context = "\n".join(d["text"] for d in retrieved)
                st.caption(f"📚 Routed to retrieval: {', '.join(d['station'] for d in retrieved)}")

            answer = generate_answer(api_key, question, context)
            st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})

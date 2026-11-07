"""Streamlit front end for the PDF RAG chatbot.

The API key, base URL and model are NEVER shown in the UI. They are read only from
  1. .streamlit/secrets.toml  (local) or Streamlit Cloud "Secrets" (deployed), or
  2. environment variables / a local .env file.
"""

import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

from rag import VectorStore, build_chunks, build_messages, extract_pages, stream_answer

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="wide")

TOP_K = 4


def load_dotenv_file() -> None:
    """Minimal .env reader so a local .env also works (no extra dependency).

    Looks next to app.py and in the current working directory. Values from the
    file replace empty or missing environment variables.
    """
    for folder in {Path(__file__).resolve().parent, Path.cwd()}:
        path = folder / ".env"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().removeprefix("export ").strip()
                value = value.strip().strip('"').strip("'")
                if value and not os.environ.get(key):
                    os.environ[key] = value


def secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


load_dotenv_file()

API_KEY = secret("LLM_API_KEY")
BASE_URL = secret("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = secret("LLM_MODEL", "openai/gpt-oss-20b")

st.session_state.setdefault("store", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed", {})


def show_sources(hits):
    with st.expander("Sources"):
        for i, h in enumerate(hits, start=1):
            st.markdown(
                f"**[{i}] {h['source']}, page {h['page']}** (similarity {h['score']})"
            )
            st.caption(h["text"])


# ---------- sidebar: documents only ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Index documents", type="primary", disabled=not uploads):
        store = VectorStore()
        indexed = {}
        with st.spinner("Reading and embedding PDFs..."):
            for f in uploads:
                try:
                    pages = extract_pages(f.getvalue())
                except Exception as exc:
                    st.error(f"Could not read {f.name}: {exc}")
                    continue
                chunks = build_chunks(pages, source=f.name)
                indexed[f.name] = {"pages": len(pages), "chunks": store.add(chunks)}
        st.session_state.store = store
        st.session_state.indexed = indexed
        st.session_state.messages = []

    for name, info in st.session_state.indexed.items():
        st.caption(f"{name}: {info['pages']} pages, {info['chunks']} chunks")
        if info["chunks"] == 0:
            st.warning(f"No text found in {name}. It may be a scanned PDF.")

    if st.button("Clear chat"):
        st.session_state.messages = []

# ---------- main ----------
st.title("📄 Chat with your PDFs")
st.caption("Retrieval-augmented answers with page-level citations.")

if not API_KEY:
    # Details go to the terminal only, never to the web page.
    here = Path(__file__).resolve().parent
    print("[config] LLM_API_KEY not found.")
    print(f"[config] app folder: {here}")
    print(f"[config] working dir: {Path.cwd()}")
    print(f"[config] .env next to app.py: {(here / '.env').is_file()}")
    print(f"[config] .streamlit/secrets.toml next to app.py: {(here / '.streamlit' / 'secrets.toml').is_file()}")
    # No settings form on purpose: the owner configures the key on the server.
    st.error(
        "This app is not configured yet. The owner needs to add LLM_API_KEY "
        "to .streamlit/secrets.toml (local) or Streamlit Cloud Secrets (deployed)."
    )
    st.stop()

if st.session_state.store is None:
    st.info("Upload one or more PDFs in the sidebar, then click **Index documents**.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("Ask a question about your documents")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    hits = st.session_state.store.query(question, k=TOP_K)
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        try:
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)
            answer = st.write_stream(
                stream_answer(client, MODEL, build_messages(question, hits, history))
            )
        except Exception:
            # Generic message so no key, URL or internal detail is exposed to visitors.
            st.error("The AI service is unavailable right now. Please try again shortly.")
            st.stop()
        show_sources(hits)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )

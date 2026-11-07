"""Core RAG logic: PDF parsing, chunking, vector retrieval and grounded answering.

Kept free of Streamlit imports so it can be tested and reused on its own.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Iterable, Iterator

import chromadb
from pypdf import PdfReader

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about the user's PDF "
    "documents. Use ONLY the numbered context passages provided. Cite the "
    "passages you used like [1] or [2][3]. If the answer is not in the context, "
    "say you could not find it in the documents instead of guessing."
)


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    index: int


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for every page that has extractable text."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping windows, preferring to break on whitespace."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # pull the cut back to the last space so words are not split
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(
    pages: Iterable[tuple[int, str]],
    source: str,
    size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for piece in chunk_text(text, size, overlap):
            chunks.append(Chunk(piece, source, page_number, len(chunks)))
    return chunks


class VectorStore:
    """Thin wrapper around an in-memory Chroma collection."""

    def __init__(self, embedding_function=None):
        self._client = chromadb.EphemeralClient()
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        # unique name so separate Streamlit sessions never share an index
        self._collection = self._client.create_collection(
            name=f"docs_{uuid.uuid4().hex}",
            metadata={"hnsw:space": "cosine"},
            **kwargs,
        )

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._collection.add(
            ids=[f"{c.source}-{c.index}-{uuid.uuid4().hex[:8]}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def query(self, question: str, k: int = 4) -> list[dict]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_texts=[question], n_results=min(k, self.count())
        )
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "score": round(1 - dist, 3),  # cosine similarity
                }
            )
        return hits


def build_messages(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] ({h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # keep the last few turns so follow-up questions make sense
    messages.extend((history or [])[-6:])
    messages.append(
        {
            "role": "user",
            "content": f"Context passages:\n{context}\n\nQuestion: {question}",
        }
    )
    return messages


def stream_answer(client, model: str, messages: list[dict]) -> Iterator[str]:
    """Yield answer text as it streams from any OpenAI-compatible API."""
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, stream=True
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content

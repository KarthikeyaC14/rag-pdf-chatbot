import hashlib
import io
import re

import pytest
from reportlab.pdfgen import canvas

from rag import VectorStore, build_chunks, build_messages, chunk_text, extract_pages


class HashEmbedding:
    """Offline, deterministic bag-of-words embedding so tests need no model download."""

    DIM = 256

    def __call__(self, input):
        vectors = []
        for text in input:
            v = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors

    def name(self):
        return "hash-embedding"

    def embed_query(self, input):
        return self(input)

    def embed_documents(self, input):
        return self(input)


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # overlap: the start of each chunk appears at the end of the previous one
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[0] in prev


def test_chunk_text_short_and_empty():
    assert chunk_text("") == []
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunk_text_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_extract_pages_and_page_numbers():
    pdf = make_pdf(["Alpha page one", "Beta page two"])
    pages = extract_pages(pdf)
    assert [p for p, _ in pages] == [1, 2]
    assert "Beta" in pages[1][1]


def test_retrieval_returns_the_right_page():
    pdf = make_pdf(
        [
            "Photosynthesis converts sunlight water and carbon dioxide into glucose.",
            "The Treaty of Versailles was signed in 1919 ending the First World War.",
            "Python lists are mutable sequences while tuples are immutable.",
        ]
    )
    store = VectorStore(embedding_function=HashEmbedding())
    added = store.add(build_chunks(extract_pages(pdf), source="notes.pdf"))
    assert added == 3

    hits = store.query("When was the Treaty of Versailles signed?", k=2)
    assert hits[0]["page"] == 2
    assert hits[0]["source"] == "notes.pdf"


def test_build_messages_contains_numbered_context_and_history():
    hits = [{"text": "Cats sleep a lot.", "source": "a.pdf", "page": 3, "score": 0.9}]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    msgs = build_messages("Do cats sleep?", hits, history)
    assert msgs[0]["role"] == "system"
    assert "[1] (a.pdf, page 3)" in msgs[-1]["content"]
    assert msgs[1:3] == history

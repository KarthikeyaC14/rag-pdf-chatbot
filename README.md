# Chat with your PDFs: a RAG chatbot

Upload PDFs, ask questions, get streamed answers that cite the exact document and page they came from.

**Live demo:** https://ragpdfchatbot-14.streamlit.app/

## How it works

```
PDF -> pypdf (text per page) -> overlapping chunks -> ChromaDB (embeddings)
                                                          |
question -> top-k similar chunks -> prompt with numbered context -> LLM (streamed) -> answer + sources
```

- **Parsing:** `pypdf` extracts text page by page so every chunk keeps its page number.
- **Chunking:** 800-character windows with 150 characters of overlap, cut on word boundaries.
- **Retrieval:** ChromaDB (in-memory, cosine similarity) with its default `all-MiniLM-L6-v2` embeddings.
- **Generation:** any OpenAI-compatible API (Groq, Gemini, OpenAI, OpenRouter, ...). The prompt tells the model to answer only from the retrieved passages and to say so when the answer is missing.
- **UI:** Streamlit chat with streaming output and an expandable "Sources" panel per answer.

## Run locally

```bash
git clone <your-repo-url> && cd rag-pdf-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then add your key (or paste it in the sidebar)
streamlit run app.py
```

Free API keys: [Groq](https://console.groq.com) or [Google AI Studio](https://aistudio.google.com).

## Tests

```bash
pytest -q
```

The tests run offline using a small hashing embedder, so they do not download any model.

## Deploy free on Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, pick the repo, set the main file to `app.py`.
3. Under **Advanced settings > Secrets**, add:
   ```toml
   LLM_API_KEY = "your-key"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   ```
4. Deploy, then paste the URL at the top of this README and on your resume.

If you would rather not spend your own key on a public demo, leave the secret out and viewers paste their own key in the sidebar.

## Limitations

- Scanned (image-only) PDFs have no extractable text. Adding OCR (e.g. `pytesseract`) is a natural next step.
- The index lives in memory per session, so it resets when the app restarts.
- Fixed-size chunking can split a table or list. Layout-aware chunking would help.

## Ideas to extend (good interview talking points)

- Swap fixed chunks for semantic or heading-aware chunking and compare answer quality.
- Add a reranker (cross-encoder) after retrieval.
- Add a small evaluation set of question/answer pairs and measure retrieval hit rate.
- Persist the index with a Chroma `PersistentClient`.

## Resume bullets (edit to match what you actually built)

**Document Q&A Chatbot (RAG)** | Python, Streamlit, ChromaDB, LLM API, pypdf
- Built a retrieval-augmented chatbot that answers questions over uploaded PDFs, with streamed responses and page-level source citations.
- Implemented page-aware chunking with overlap, vector search in ChromaDB, and a grounded prompt that refuses to answer outside the retrieved context.
- Added unit tests for chunking and retrieval and deployed the app on Streamlit Community Cloud.

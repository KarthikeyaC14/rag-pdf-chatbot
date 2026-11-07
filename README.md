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
4. Deploy, then paste the URL at the top of this README.

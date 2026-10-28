# Simple Agent

Ask questions about your own documents. Upload PDF, DOCX, TXT or Markdown files, and the assistant answers using only what it finds in them, citing the source file (and page number for PDFs).

Built with [LangChain](https://python.langchain.com/), [OpenRouter](https://openrouter.ai/) (chat and embeddings), [Chroma](https://www.trychroma.com/) (local vector store) and [Streamlit](https://streamlit.io/) (web UI).

## How it works

1. **Ingest:** files are loaded, split into 1,000-character chunks (200 overlap), embedded, and stored in a local Chroma index. Re-uploading the same file doesn't create duplicates.
2. **Condense:** on follow-up questions, the model rewrites your question into a standalone search query using the chat history.
3. **Retrieve:** the top 4 relevant chunks are fetched with MMR search, which reduces near-duplicate results.
4. **Answer:** the model answers from those chunks only. If the answer isn't in your documents, it says so instead of guessing.

## Project structure

```
.
├── app.py             # Streamlit chat UI
├── assistant.py       # Core RAG logic + command-line chat
├── requirements.txt
├── .env.example       # Template for your settings
└── chroma_db/         # Created on first run (the vector index)
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create your settings file and add your key from [openrouter.ai/keys](https://openrouter.ai/keys):

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=sk-or-...
MODEL=openai/gpt-5.6-luna
```

Your OpenRouter account needs credits for the models you choose.

## Usage

### Web UI

```bash
streamlit run app.py
```

Open http://localhost:8501, upload files in the sidebar, and start asking questions. Use **Clear index & chat** to wipe the index and start over.

### Command line

```bash
python assistant.py report.pdf notes.md   # ingest files, then chat
python assistant.py                       # chat over previously indexed files
```

## Configuration

Settings are read from `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `MODEL` | `poolside/laguna-s-2.1:free` | Chat model. OpenRouter slugs include the provider prefix |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Override the API endpoint |

Browse available model names at [openrouter.ai/models](https://openrouter.ai/models).

Retrieval depth (`k`), chunk size and overlap can be changed in `assistant.py`. To support more file types, add a loader to the `LOADERS` dict.

## Troubleshooting

- **Blank page in the browser:** you ran `streamlit run assistant.py`. Run `streamlit run app.py` instead.
- **Model not found:** check that `MODEL` exactly matches a slug on openrouter.ai/models, including the provider prefix.
- **Error mentioning `platform.openai.com` or a 429 "no credits":** the app is still calling OpenAI. Make sure you're using the OpenRouter version of `assistant.py` and restart Streamlit.
- **Poor or failing results after changing the embedding model:** delete the `chroma_db/` folder (or click **Clear index & chat**) and re-upload your files. Vectors from different embedding models can't be mixed.
- **Scanned PDFs return nothing:** only embedded text is extracted. Run OCR on the file first.

## Security

`.env` contains your API key. Keep it out of version control by listing it in `.gitignore`.

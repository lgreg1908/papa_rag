# PapaRAG Document Explorer

A local Retrieval‑Augmented Generation (RAG) app to explore, search, and ask questions over your own documents by uploading a ZIP archive.
It's called 'PapaRag' because it was built primariy to help my father - investigative journalist - navigate the thousands of documents collected in decades of work.

- **Streamlit UI** (`src/app.py`):  
  - Upload a ZIP of your document folder  
  - Unzip, preprocess, embed and persist a FAISS index (`data/vector_store.faiss` + `data/metadata.pkl`)  
  - Ask free‑text questions, retrieve top‑K chunks, display answer + source snippets  

- **Ingestion** (`src/ingestion/loader.py`):  
  - Recursive file discovery & document loading (PDF, DOCX, TXT, MD)  

- **Preprocessing** (`src/processing/preprocess.py`):  
  - `normalize_documents()`: clean whitespace and non‑printables  
  - `chunk_documents()`: split text into overlapping chunks + tag each chunk  

- **Embeddings** (`src/processing/embeddings.py`):  
  - `get_text_embeddings()`: OpenAI embeddings API with in‑memory cache  
  - `embed_documents()`: attach embeddings to each chunk  

- **Vector Store** (`src/retrieval/vector_store.py`):  
  - `FaissVectorStore`: persists embeddings, metadata, and snippet text  
  - `search()`: returns nearest neighbors + distances  

- **Q&A** (`src/qa/qa.py`):  
  - `answer_question()`: assemble prompt with retrieved snippets + call OpenAI ChatCompletion  

- **Scoring** (`src/utils/scoring.py`):  
  - `distance_to_score()`: convert FAISS L2 distance into a 0–100 relevance score  

## Getting Started

### 1. Clone and enter the repo
```bash
git clone https://github.com/lgreg1908/papa_rag.git
cd papa_rag
```

### 2. Install dependencies
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_COMPLETION_MODEL=gpt-3.5-turbo
MODEL_MAX_TOKENS=4096
RESERVED_TOKENS=512
```

### 3. Run locally with Streamlit
```bash
streamlit run src/app.py
```

- Upload a ZIP archive of your document folder in the sidebar
- Wait for indexing (persisted to `data/vector_store.faiss` + `data/metadata.pkl`)
- Ask questions in the main panel

### 4. Run via Docker
Build and run with Docker Compose:

```bash
docker-compose up --build
```

Then visit `http://localhost:8501`.

### 5. Run tests
```bash
pytest -q
```

## Next Steps
- **RetrievalQA chain:** integrate LangChain’s RetrievalQA for better answer formatting and citation.

- **Conversation memory:** add per‑session history for multi‑turn interaction.

- **UI improvements:** progress indicators, error handling, and mobile support.

- **Packaging:** publish as a Docker image on Docker Hub or as a Python package.

- **Keyword fallback:** add a Whoosh or SQLite full‑text index for pure keyword search fallback.

- **Diff/tagging viewer:** build side‑by‑side diff and tagging components for document review.

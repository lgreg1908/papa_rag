# PapaRAG Document Explorer

A local **RAG-enabled** (Retrieval-Augmented Generation) application to **explore, search, and interrogate** documents and images in any folder on your machine. It's called 'PapaRag' because it was built primariy to help my father - investigative journalist - navigate the thousands of documents collected in decades of work.

**End Goal:**  
- **Real-time ingestion** of documents (PDF, DOCX, TXT, MD).  
- **Preprocessing**: normalize text, extract metadata (timestamps, char counts).  
- **Embedding**: generate semantic vectors for text (OpenAI) and images (CLIP), with caching.  
- **Vector store & retrieval**: FAISS-based nearest-neighbor search + full-text fallback.  
- **RAG & QA**: build RetrievalQA chains over your personal document corpus.  
- **Tagging & Diffing**: auto-tag with LLM; side-by-side text and image comparisons.  
- **Streamlit UI**: intuitive folder browser, live ingestion, search interface, tag editor.  

---

## Current Version

We’ve built the core ingestion and embedding pipeline, plus a minimal Streamlit frontend. Currently only works with text data.

### Ingestion
- **`src/ingestion/loader.py`** —  
  • `list_supported_files(folder_path)`  
  • `load_documents(paths)` → LangChain `Document` objects  
  • `load_folder(folder_path)` → `docs`

- **`src/ingestion/watcher.py`** —  
  • `FolderWatcher(folder, callback)` watches and triggers on-create/on-modify events  
  • `IngestionHandler` processes single files and calls your callback  

### Processing
- **`src/processing/preprocess.py`** —  
  • `normalize_documents(docs)`  
  • `chunk_documents(docs, chunk_size, chunk_overlap)`  

- **`src/processing/embeddings.py`** —  
  • `get_text_embeddings(texts)` (OpenAI v1 SDK + cache)  
  • `embed_documents(docs)`  

### Retrieval
- **`src/retrieval/vector_store.py`** —  
  • `FaissVectorStore(index_path, meta_path)`

### Frontend
- **`src/app.py`** (Streamlit) —  
  • Folder browser in sidebar  
  • “Connect” button lists & loads supported files  
  • Displays counts of document chunks & images loaded

### Testing & CI
- **Pytest** covers loader, watcher, preprocess, embeddings.  
- **GitHub Actions** runs tests on PRs (Python 3.10–3.12).  

---

## Getting Started

1. **Clone & enter** the repo:  
   ```bash
   git clone https://github.com/lgreg1908/papa_rag.git
   cd rag_app
   ```

2. **Create & activate** a Python 3.10 virtual environment:  
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
   
3. **Install dependencies:**:  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure:** OpenAI API key into `.env` file:  
   ```dotenv
    OPENAI_API_KEY=your_api_key_here
    # (optional) OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
    ```

5. **Run the App**:
    ```bash 
    streamlit run src/app.py
    ```
    - Use the sidebar to browse to a folder of PDFs, DOCX, images, etc.
    - Click **Connect** to load and view counts of your files.

6. **Run demos**:
    > **Note:** Before running the demos, place your sample PDFs, Word docs, text files, and images into the `data/sample` folder. By default, the folder sample documents related to a fantasy corruption case.

    __Individual modules__

    ```bash

    # Loader demo (loads the sample documents)
    python -m src.ingestion.loader

    # Preprocessing demo (normalizes text)
    python -m src.processing.preprocess

    # Embeddings demo (generates text embeddings)
    python -m src.processing.embeddings

    ```

    __Full pipeline__
    ```bash

    # Create vector store
    python -m src.main ingest <folder-name>

    # Start the watcher
    python -m src.main watch <folder-name>

    # Query the store
    python -m src.main search <query> --top_k <top-k>

    ```

7. **Run tests**:
    ```bash
    pytest -q
    ```

## Next Steps

- **RAG & QA**: Integrate LangChain’s RetrievalQA chains to answer questions over your document corpus.
- **Tagging & Diff Viewer**: Implement `src/tagging/tagger.py` for LLM-driven auto-tagging and manual tag editing, plus side-by-side diff views.
- **UI Enhancements**: Extend the Streamlit app with search input, result lists (snippets + thumbnails), tag filters, and a Q&A panel.
- **Packaging**: Create a `pyproject.toml` or `setup.py` for `pip install -e .`, and refine the Docker image for distribution.

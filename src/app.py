#!/usr/bin/env python3
import streamlit as st
import tempfile
import zipfile
import os
from dotenv import load_dotenv

from src.ingestion.loader import load_folder
from src.processing.preprocess import normalize_documents, chunk_documents
from src.processing.embeddings import embed_documents, get_text_embeddings
from src.retrieval.vector_store import FaissVectorStore
from src.qa.qa import answer_question
from src.utils.scoring import distance_to_score

# Constants
MAX_CHUNKS_DEFAULT = 20
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
RESERVED_TOKENS = int(os.getenv("RESERVED_TOKENS", "512"))
AVERAGE_TOKEN_RATIO = 4  # approx chars per token

load_dotenv()

@st.cache_resource(show_spinner=False)
def load_store(index_path: str, meta_path: str) -> FaissVectorStore:
    """
    Cached resource: loads or initializes the FAISS store from disk.
    """
    return FaissVectorStore(index_path=index_path, meta_path=meta_path)


def configure_page():
    st.set_page_config(page_title="RAG Q&A Explorer", layout="wide")
    st.title("📚 PapaRAG - Document Q&A Explorer")


def init_state():
    """
    Initialize session state and reload FAISS store if files exist.
    Also compute how many chunks are in the loaded index.
    """
    if 'store' not in st.session_state:
        idx_path, meta_path = "data/vector_store.faiss", "data/metadata.pkl"
        st.session_state.store = load_store(idx_path, meta_path)

    if 'max_k' not in st.session_state:
        st.session_state.max_k = MAX_CHUNKS_DEFAULT

    # Once we have a store object, expose chunk count for UI
    if st.session_state.store.index is not None:
        # metadata_list holds one entry per chunk
        st.session_state.current_chunk_count = len(st.session_state.store.metadata_list)
    else:
        st.session_state.current_chunk_count = 0


def zip_upload_panel():
    """Sidebar form: upload a ZIP of your docs and index once."""
    with st.sidebar.form("zip_form"):
        st.header("📤 Upload & Index (zip)")
        zip_file = st.file_uploader(
            "Drag & drop a ZIP of your folder",
            type="zip"
        )
        submit = st.form_submit_button("Unzip & Index")

    if submit and zip_file:
        with st.spinner("Unzipping & indexing…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                # unpack
                with zipfile.ZipFile(zip_file) as z:
                    z.extractall(tmpdir)
                # pipeline
                raw = load_folder(tmpdir)
                norm = normalize_documents(raw)
                chunks = chunk_documents(norm)
                emb = embed_documents(chunks)

                # build + persist store
                store = st.session_state.store
                store.delete()
                store.add_documents(emb)
                # update count
                st.session_state.current_chunk_count = len(emb)

                # recompute slider max
                if chunks:
                    avg_chars = sum(len(d.page_content) for d in chunks) / len(chunks)
                    avg_tokens = avg_chars / AVERAGE_TOKEN_RATIO
                    max_k = int((MODEL_MAX_TOKENS - RESERVED_TOKENS) / avg_tokens)
                    st.session_state.max_k = max(1, max_k)
                else:
                    st.session_state.max_k = MAX_CHUNKS_DEFAULT

        st.sidebar.success(f"✅ Indexed {len(emb)} chunks from ZIP!")


def sidebar_status_panel():
    """Show current index status (persisted)."""
    with st.sidebar:
        count = st.session_state.current_chunk_count
        if count > 0:
            st.success(f"⚡ Loaded existing index with **{count}** chunks.")
        else:
            st.info("No index loaded yet.")


def qa_panel():
    """Main Q&A UI once store is ready."""
    store = st.session_state.store
    if store.index is None:
        st.info("📋 Please upload a ZIP of your documents in the sidebar.")
        return

    st.subheader("🤖 Ask a Question")
    question = st.text_input("Enter your question here:")
    top_k = st.slider("How many chunks to retrieve?", 1, st.session_state.max_k, 1)

    if st.button("💬 Ask") and question.strip():
        # 1) embed query
        qvec = get_text_embeddings([question])[0]
        # 2) search
        docs, dists = store.search(qvec, top_k)
        # 3) score
        scores = [distance_to_score(d, max_distance=2.0) for d in dists]
        # 4) answer
        answer, used = answer_question(question, docs)

        st.markdown("### 📝 Answer")
        st.write(answer)

        with st.expander(f"📑 Sources ({len(used)})", expanded=False):
            for doc, dist, score in zip(used, dists, scores):
                cid = doc.metadata.get("chunk_id", "<unknown>")
                src = doc.metadata.get("source", "<unknown>")
                snippet = doc.page_content.replace("\n", " ")
                st.markdown(
                    f"- **Chunk:** `{cid}`  \n"
                    f"  • **Source:** `{src}`  \n"
                    f"  • **Distance:** {dist:.4f}  \n"
                    f"  • **Relevance:** {score:.1f}/100  \n"
                )


def main():
    configure_page()
    init_state()
    zip_upload_panel()
    sidebar_status_panel()
    qa_panel()


if __name__ == "__main__":
    main()

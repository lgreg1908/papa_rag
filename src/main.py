#!/usr/bin/env python3
import argparse
from langchain.schema import Document

from src.ingestion.loader import load_folder
from src.processing.preprocess import normalize_documents, chunk_documents
from src.processing.embeddings import embed_documents, get_text_embeddings
from src.retrieval.vector_store import FaissVectorStore
from src.qa.qa import answer_question
from src.utils.scoring import distance_to_score


def build_index(folder: str, text_store: FaissVectorStore) -> None:
    """
    One-shot ingest: load, preprocess, embed, and index all files in `folder` into a vector store.
    """
    print(f"[INGEST] Scanning folder: {folder}")
    raw_docs: list[Document] = load_folder(folder)
    print(f"[INGEST] Loaded {len(raw_docs)} document chunks.")

    # Preprocess & embed text
    docs_norm: list[Document] = normalize_documents(raw_docs)
    docs_chunked: list[Document] = chunk_documents(docs_norm, chunk_size=250, chunk_overlap=50)
    docs_emb: list[Document] = embed_documents(docs_chunked)

    # Create vector store
    text_store.add_documents(docs_emb)
    print(f"[INGEST] Indexed {len(docs_emb)} chunks from {len(docs_norm)} documents")


def search_text(query: str, text_store: FaissVectorStore, top_k: int) -> None:
    """Search text-only index."""
    print(f"[SEARCH] Query: '{query}' (top_k={top_k})")
    vec = get_text_embeddings([query])[0]
    results, dists = text_store.search(vec, top_k)
    for i, (doc, dist) in enumerate(zip(results, dists), start=1):
        print(f"{i}. {doc.metadata.get('chunk_id')}. {dist}")

def run_qa(query: str, text_store: FaissVectorStore, top_k: int) -> None:
    """
    Retrieve top‐k chunks, call the LLM, and display answer + sources.
    """
    print(f"[QA] Question: \"{query}\" (top_k={top_k})\n")
    vec = get_text_embeddings([query])[0]
    docs, dists = text_store.search(vec, top_k)
    answer, used = answer_question(query, docs)
    print("=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    for doc, dist in zip(used, dists):
        cid = doc.metadata.get("chunk_id", "<chunk>")
        print(f"- {cid} - relevance: {distance_to_score(dist, max_distance=2, min_score=0, max_score=1)}, distance:{dist}")

def reset_index(text_store: FaissVectorStore) -> None:
    """
    Delete on-disk index and metadata to start fresh.
    """
    text_store.delete()
    print("[RESET] FAISS index and metadata cleared.")

def main():
    parser = argparse.ArgumentParser(description="RAG Document Explorer CLI with separate text/image indices")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ingest
    p_ingest = sub.add_parser('ingest', help='One-shot ingest of a folder')
    p_ingest.add_argument('folder', help='Folder to ingest')

    # search-text
    p_st = sub.add_parser('search', help='Search by text query')
    p_st.add_argument('query', help='Text query')
    p_st.add_argument('--top_k', type=int, default=5)

    # qa
    p_qa = sub.add_parser("qa", help="Ask a question over indexed documents")
    p_qa.add_argument("question", help="Natural language question")
    p_qa.add_argument("--top_k", type=int, default=5)

    # reset
    sub.add_parser("reset", help="Delete the FAISS index and metadata files")

    args = parser.parse_args()

    # Initialize separate stores
    store = FaissVectorStore(index_path='data/vector_store.faiss', meta_path='data/metadata.pkl')

    if args.cmd == 'ingest':
        build_index(args.folder, store)
    elif args.cmd == 'search':
        search_text(args.query, store, args.top_k)
    elif args.cmd == "qa":
        run_qa(args.question, store, args.top_k)
    elif args.cmd == "reset":
        reset_index(store)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

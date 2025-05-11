#!/usr/bin/env python3
import argparse
import sys
import os

from PIL import Image
from langchain.schema import Document

from src.ingestion.loader import load_folder
from src.ingestion.watcher import FolderWatcher
from src.processing.preprocess import normalize_documents
from src.processing.embeddings import embed_documents, get_text_embeddings
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.retriever import Retriever


def build_index(folder: str, text_store: FaissVectorStore) -> None:
    """
    One-shot ingest: load, preprocess, embed, and index all files in `folder` into a vector store.
    """
    print(f"[INGEST] Scanning folder: {folder}")
    raw_docs = load_folder(folder)
    print(f"[INGEST] Loaded {len(raw_docs)} document chunks.")

    # Preprocess & embed text
    docs_norm = normalize_documents(raw_docs)
    docs_emb = embed_documents(docs_norm)
    text_store.add_documents(docs_emb)
    print(f"[INGEST] Indexed {len(docs_emb)} text documents")


def start_watcher(folder: str, text_store: FaissVectorStore) -> None:
    """
    Watch a folder and auto-index new or modified files into separate stores.
    """
    def callback(docs):
        docs_norm = normalize_documents(docs)
        docs_emb = embed_documents(docs_norm)
        text_store.add_documents(docs_emb)
        print(f"[WATCH] Indexed {len(docs_emb)} text chunks")

    watcher = FolderWatcher(folder, callback)
    print(f"[WATCH] Monitoring {folder} (Ctrl+C to stop)...")
    watcher.run()


def search_text(query: str, text_store: FaissVectorStore, top_k: int) -> None:
    """Search text-only index."""
    print(f"[SEARCH] Query: '{query}' (top_k={top_k})")
    vec = get_text_embeddings([query])[0]
    results = text_store.search(vec, top_k)
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc.metadata.get('source')} on text index")


def main():
    parser = argparse.ArgumentParser(description="RAG Document Explorer CLI with separate text/image indices")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ingest
    p_ingest = sub.add_parser('ingest', help='One-shot ingest of a folder')
    p_ingest.add_argument('folder', help='Folder to ingest')

    # watch
    p_watch = sub.add_parser('watch', help='Watch a folder and index changes')
    p_watch.add_argument('folder', help='Folder to watch')

    # search-text
    p_st = sub.add_parser('search', help='Search by text query')
    p_st.add_argument('query', help='Text query')
    p_st.add_argument('--top_k', type=int, default=5)

    args = parser.parse_args()

    # Initialize separate stores
    text_store = FaissVectorStore(index_path='data/vector_store.faiss', meta_path='data/metadata.pkl')

    if args.cmd == 'ingest':
        build_index(args.folder, text_store)
    elif args.cmd == 'watch':
        start_watcher(args.folder, text_store)
    elif args.cmd == 'search':
        search_text(args.query, text_store, args.top_k)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

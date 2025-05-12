import os
import shutil
from typing import List

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from langchain.schema import Document

def build_whoosh_index(
    index_dir: str,
    docs: List[Document],
    content_field: str = "content",
    path_field: str = "path"
) -> None:
    """
    (Re)Creates a Whoosh index under `index_dir` from a list of pre-normalized Documents.

    Args:
      index_dir: Path where Whoosh will store its index files.
      docs: List of LangChain Documents with .page_content and metadata['source'] set.
      content_field: Name of the TEXT field in the schema.
      path_field: Name of the stored ID field pointing to each document’s source path.
    """
    # 1. Define schema
    schema = Schema(
        **{
            path_field: ID(stored=True, unique=True),
            content_field: TEXT
        }
    )

    # 2. Wipe + recreate index directory
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)

    # 3. Create index and writer
    ix = index.create_in(index_dir, schema)
    writer = ix.writer()

    # 4. Index each document chunk
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("file_path")
        writer.add_document(**{
            path_field: src,
            content_field: doc.page_content
        })

    writer.commit()


def main() -> None:
    """
    Demo: load & normalize documents, then build and query a Whoosh index.
    """
    from whoosh import index as whoosh_idx
    from whoosh.qparser import QueryParser
    from src.ingestion.loader import list_supported_files, load_documents
    from src.processing.preprocess import normalize_documents

    index_dir = "data/whoosh_index"
    folder = "data/sample"

    # 1) Load & preprocess
    print(f"Loading files from '{folder}'…")
    paths = list_supported_files(folder)
    raw_docs = load_documents(paths)
    docs = normalize_documents(raw_docs)
    print(f"Prepared {len(docs)} document chunks for indexing.")

    # 2) Build Whoosh index
    print(f"Building Whoosh index at '{index_dir}'…")
    build_whoosh_index(index_dir=index_dir, docs=docs)
    print("Index built.\n")

    # 3) Open index and run sample queries
    ix = whoosh_idx.open_dir(index_dir)
    searcher = ix.searcher()
    parser = QueryParser("content", schema=ix.schema)

    for query_text in ["kickback", "machine learning", "streamlit"]:
        print(f"Results for '{query_text}':")
        q = parser.parse(query_text)
        hits = searcher.search(q, limit=5)
        if hits:
            for hit in hits:
                print(f" - {hit['path']} (score: {hit.score:.2f})")
        else:
            print(" - no hits found")
        print()

    # 4) Cleanup
    searcher.close()
    shutil.rmtree(index_dir)
    print("Cleaned up index directory.")


if __name__ == "__main__":
    main()

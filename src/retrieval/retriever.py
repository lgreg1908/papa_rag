import os
from typing import List, Optional
from langchain.schema import Document

from src.retrieval.vector_store import FaissVectorStore
from src.processing.embeddings import get_text_embeddings, embed_documents

# Optional Whoosh import
try:
    from whoosh import index as whoosh_index
    from whoosh.qparser import QueryParser
except ImportError:
    whoosh_index = None
    QueryParser = None


class Retriever:
    """
    High-level retriever wrapping FaissVectorStore, supporting text or image queries,
    with optional Whoosh keyword fallback.
    """
    def __init__(
        self,
        vector_store: FaissVectorStore,
        whoosh_index_dir: Optional[str] = None,
        whoosh_field: str = 'content'
    ):
        """
        Args:
            vector_store: Initialized FaissVectorStore instance.
            whoosh_index_dir: Path to an existing Whoosh index directory for fallback.
            whoosh_field: Name of the text field in Whoosh schema.
        """
        self.vector_store = vector_store
        self.whoosh_searcher = None
        if whoosh_index_dir and whoosh_index and os.path.isdir(whoosh_index_dir):
            idx = whoosh_index.open_dir(whoosh_index_dir)
            self.whoosh_searcher = idx.searcher()
            self.whoosh_parser = QueryParser(whoosh_field, schema=idx.schema)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fallback: bool = True
    ) -> List[Document]:
        """
        Retrieve top_k Documents for a text or image query.

        Args:
            query: Input query, either a text string or PIL Image.
            top_k: Number of hits to return.
            fallback: Whether to use Whoosh fallback if FAISS returns fewer than top_k.

        Returns:
            List of LangChain Document objects (metadata-only).
        """
        # Embed query
        if isinstance(query, str):
            vec = get_text_embeddings([query])[0]
        else:
            raise ValueError("Query must be text ")

        # FAISS search
        results = self.vector_store.search(vec, top_k)

        # Fallback to Whoosh if needed
        if fallback and self.whoosh_searcher and isinstance(query, str) and len(results) < top_k:
            needed = top_k - len(results)
            qs = self.whoosh_parser.parse(query)
            hits = self.whoosh_searcher.search(qs, limit=needed)
            for hit in hits:
                # Assume schema has 'path' field for document path
                meta = {'source': hit.get('path'), 'score': hit.score}
                results.append(Document(page_content='', metadata=meta))
        return results

    def close(self) -> None:
        """
        Close any resources (e.g., Whoosh searcher).
        """
        if self.whoosh_searcher:
            self.whoosh_searcher.close()

def main() -> None:
    """
    Real-embedding demo: index and search sample text documents using OpenAI embeddings.
    """
    from src.ingestion.loader import load_folder
    from src.processing.preprocess import normalize_documents

    # Initialize FAISS store
    idx_path = 'data/test_store.faiss'
    meta_path = 'data/test_meta.pkl'
    store = FaissVectorStore(index_path=idx_path, meta_path=meta_path)
    store.delete()
    print("Cleared existing FAISS store for real embeddings.")

    # Load and clean the source files
    folder_path = 'data/sample'
    docs: list[Document] = load_folder(folder_path)
    docs_norm: list[Document] = normalize_documents(docs)

    # Embed and add to store
    docs_emb: list[Document] = embed_documents(docs_norm)
    store.add_documents(docs_emb)
    print(f"Indexed {len(docs_emb)} text documents.")

    # Create retriever
    retriever = Retriever(store)

    # Text query
    query = "payment allocation"
    print(f"\nSearching for query: '{query}'")
    results = retriever.retrieve(query, top_k=3)
    for i, doc in enumerate(results, start=1):
        print(f"{i}. source={doc.metadata.get('source')} | embedding length={len(doc.metadata.get('embedding', []))}")

    # Cleanup
    retriever.close()
    store.delete()
    print("Removed demo index files.")

if __name__ == '__main__':
    main()

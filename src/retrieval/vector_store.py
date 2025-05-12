import os
import pickle
from typing import List, Optional, Tuple
import faiss
import numpy as np
from langchain.schema import Document


class FaissVectorStore:
    """
    A simple FAISS-backed vector store for LangChain Documents.

    Persists index, metadata, and text snippets to disk, supports storing and retrieving
    embeddings along with per-document metadata, text content, and retrieval distances.
    """
    def __init__(
        self,
        index_path: str = 'data/embeddings.faiss',
        meta_path: str = 'data/metadata.pkl'
    ):
        """
        Args:
            index_path: Path to store the FAISS index file.
            meta_path: Path to store the pickled metadata list.
        """
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: Optional[faiss.Index] = None
        self.metadata_list: List[dict] = []
        self._load()

    def _load(self) -> None:
        """
        Load existing index and metadata from disk, if available.
        Otherwise, initialize empty state.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                self.metadata_list = pickle.load(f)
        else:
            self.index = None
            self.metadata_list = []

    def _save(self) -> None:
        """
        Persist the FAISS index and metadata list to disk.
        """
        for path in (self.index_path, self.meta_path):
            dirpath = os.path.dirname(path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata_list, f)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents (must have 'embedding' in metadata) to the index.

        Args:
            docs: List of LangChain Document objects with precomputed embeddings
                  and page_content carrying the text snippet.
        """
        if not docs:
            return

        vectors = []
        for doc in docs:
            vec = doc.metadata.get('embedding')
            if vec is None:
                raise ValueError("Document missing 'embedding' in metadata")
            vectors.append(vec)
            # Build metadata entry: include all original metadata and snippet text
            entry = dict(doc.metadata)
            entry['page_content'] = doc.page_content
            self.metadata_list.append(entry)

        arr = np.array(vectors, dtype='float32')

        if self.index is None:
            dim = arr.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(arr)
        self._save()

    def search(
        self,
        query_embeddings: List[float],
        top_k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """
        Perform a nearest‐neighbor search against the index.

        Returns a tuple:
        - List of Documents (with page_content and original metadata, sans distance)
        - Parallel list of float distances for each hit
        """
        if self.index is None:
            return [], []

        vec = np.array(query_embeddings, dtype='float32').reshape(1, -1)
        distances, indices = self.index.search(vec, top_k)

        docs: List[Document]   = []
        dists: List[float]     = []

        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata_list):
                entry = dict(self.metadata_list[idx])
                snippet = entry.pop('page_content', '')
                docs.append(Document(page_content=snippet, metadata=entry))
                dists.append(float(dist))

        return docs, dists

    def delete(self) -> None:
        """
        Remove on-disk index and metadata, and reset in-memory state.
        """
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        self.index = None
        self.metadata_list = []

def main() -> None:
    """
    Demo of the FaissVectorStore with snippet persistence and distance outputs.
    """
    from langchain.schema import Document

    # initialize & clear any existing demo store
    store = FaissVectorStore(
        index_path="data/embeddings_demo.faiss",
        meta_path="data/metadata_demo.pkl"
    )
    store.delete()
    print("Cleared existing FAISS demo store.")

    # create 5 synthetic documents with simple 2-D embeddings
    docs = []
    for i in range(5):
        vec = [float(i), float(5 - i)]
        docs.append(
            Document(
                page_content=f"Document {i} content here.",
                metadata={
                    "source": f"doc_{i}.txt",
                    "embedding": vec
                }
            )
        )

    # add to the store
    store.add_documents(docs)
    print(f"Indexed {len(docs)} demo documents.\n")

    # perform a search near [0.0, 5.0]
    query_vec = [0.0, 5.0]
    results, distances = store.search(query_vec, top_k=3)

    print(f"Top 3 results for query vector {query_vec}:")
    for rank, (doc, dist) in enumerate(zip(results, distances), start=1):
        src = doc.metadata.get("source", "<unknown>")
        snippet = doc.page_content
        print(
            f"{rank}. source={src}  distance={dist:.4f}\n"
            f"    snippet: {snippet!r}\n"
        )

    # cleanup demo files
    os.remove("data/embeddings_demo.faiss")
    os.remove("data/metadata_demo.pkl")
    print("Cleaned up demo store files.")

if __name__ == "__main__":
    main()

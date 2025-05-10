import os
import pickle
from typing import List, Optional
import faiss
import numpy as np
from langchain.schema import Document


class FaissVectorStore:
    """
    A simple FAISS-backed vector store for LangChain Documents.

    Persists index and metadata to disk.
    """
    def __init__(
            self, 
            index_path: str='data/embeddings.faiss',
            meta_path: str='data/metadata.pkl'
        ):
        """
        Args:
            index_path: Path to store the FAISS index file.
            meta_path: Path to store the pickled metadata list.
        """

        self.index_path = index_path
        self.meta_path = meta_path
        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []
        self._load()

    def _load(self) -> None:
        """
        Load existing index and metadata from disk, if available.
        Otherwise, initialize empty state.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f) 
        
        else:
            self.index = None
            self.metadata = []
    
    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents (must have 'embedding' in metadata) to the index.

        Args:
            docs: List of LangChain Document objects with precomputed embeddings.
        """
        if not docs:
            return 
        
        # Extract vectors and metadata
        vectors = []
        for doc in docs:
            vec = doc.metadata.get('embedding')
            if vec is None:
                raise ValueError("Document missing 'embedding' in metadata")
            vectors.append(vec)
            self.metadata.append(dict(doc.metadata))
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
    ) -> List[Document]:
        """
        Perform a nearest-neighbor search against the index.

        Args:
            query_embedding: Single embedding vector (list of floats).
            top_k: Number of nearest neighbors to return.

        Returns:
            List of Document objects reconstructed from metadata (no content).
        """
        if self.index is None:
            return []
        
        vec = np.array(query_embeddings, dtype='float32').reshape(1, -1)
        _, indices = self.index.search(vec, top_k)
        results: List[Document]=[]
        for idx in indices[0]:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(Document(page_content='', metadata=meta))
        return results

    def _save(self) -> None:
        """
        Persist the FAISS index and metadata list to disk.
        """
        # Check if dirs exist
        for path in (self.index_path, self.meta_path):
            dirpath = os.path.dirname(path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def delete(self) -> None:
        """
        Remove on-disk index and metadata, and reset in-memory state.
        """
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        self.index = None
        self.metadata = []

def main() -> None:
    """
    Demo of the FaissVectorStore:
      - Deletes any existing store
      - Adds a few sample Documents with synthetic embeddings
      - Performs a nearest-neighbor search
      - Prints out the matching metadata
    """
    import numpy as np
    from langchain.schema import Document

    # Initialize and reset the store
    store = FaissVectorStore(index_path="data/embeddings_test.faiss",
                             meta_path="data/metadata_test.pkl")
    store.delete()
    print("Cleared existing index and metadata.")

    # Create sample documents with 2-D embeddings
    docs = []
    for i in range(5):
        vec = [float(i), float(5 - i)]   # simple 2D vectors
        docs.append(Document(
            page_content=f"Document {i}",
            metadata={"source": f"doc_{i}.txt", "embedding": vec}
        ))

    store.add_documents(docs)
    print(f"Added {len(docs)} documents to the store.")

    # Run a search for a query near [0, 5]
    query_vec = [0.0, 5.0]
    results = store.search(query_vec, top_k=3)
    print(f"\nTop 3 results for query {query_vec}:")
    for rank, doc in enumerate(results, start=1):
        print(f"{rank}. source={doc.metadata['source']}, embedding={doc.metadata['embedding']}")
    os.remove('data/embeddings_test.faiss')
    os.remove('data/metadata_test.pkl')

if __name__ == "__main__":
    main()
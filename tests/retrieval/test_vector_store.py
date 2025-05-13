import os
import pytest
from pathlib import Path
from langchain.schema import Document

from src.retrieval.vector_store import FaissVectorStore


def make_store(tmp_path: Path):
    """Helper to create a FaissVectorStore with temp paths."""
    index_path = str(tmp_path / "test_index.faiss")
    meta_path = str(tmp_path / "test_meta.pkl")
    return FaissVectorStore(index_path=index_path, meta_path=meta_path)


def test_search_empty(tmp_path):
    store = make_store(tmp_path)
    # No data added — search returns empty lists
    docs, dists = store.search([0.0, 1.0], top_k=3)
    assert docs == []
    assert dists == []


def test_add_and_search(tmp_path):
    store = make_store(tmp_path)
    store.delete()  # ensure clean slate

    # Prepare sample documents with simple 2D embeddings
    docs_in = []
    for i in range(5):
        vec = [float(i), float(5 - i)]
        docs_in.append(Document(
            page_content=f"doc_{i}",
            metadata={"source": f"doc_{i}", "embedding": vec}
        ))

    store.add_documents(docs_in)

    # After adding, files should exist
    assert os.path.exists(store.index_path)
    assert os.path.exists(store.meta_path)

    # Search near [0,5] should return doc_0 first
    docs_out, dists = store.search([0.0, 5.0], top_k=3)
    assert len(docs_out) == 3
    # Check that the first result's metadata matches doc_0
    assert docs_out[0].metadata["source"] == "doc_0"
    # Distances should be a list of floats
    assert isinstance(dists, list)
    assert all(isinstance(d, float) for d in dists)


def test_missing_embedding_raises(tmp_path):
    store = make_store(tmp_path)
    store.delete()
    # Document without 'embedding' key
    doc = Document(page_content="x", metadata={"source": "b"})
    with pytest.raises(ValueError):
        store.add_documents([doc])


def test_persistence(tmp_path):
    store = make_store(tmp_path)
    store.delete()

    # Add one document
    doc = Document(
        page_content="x",
        metadata={"source": "a", "embedding": [1.0, 2.0]}
    )
    store.add_documents([doc])

    # New instance should load existing index and metadata
    new_store = make_store(tmp_path)
    # metadata_list should be loaded on init
    assert len(new_store.metadata_list) == 1

    # Search returns the same document metadata
    docs_out, dists = new_store.search([1.0, 2.0], top_k=1)
    assert len(docs_out) == 1
    assert docs_out[0].metadata["source"] == "a"
    assert len(dists) == 1
    assert pytest.approx(dists[0]) == 0.0
    
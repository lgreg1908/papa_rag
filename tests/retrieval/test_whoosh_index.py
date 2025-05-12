import os
import shutil
import pytest

from whoosh import index as whoosh_index
from whoosh.qparser import QueryParser
from langchain.schema import Document

from src.retrieval.whoosh_index import build_whoosh_index


def make_docs() -> list[Document]:
    # Create sample Document objects with distinct content and source
    return [
        Document(page_content="Hello world from file one", metadata={"source": "file1.txt"}),
        Document(page_content="This is the second document", metadata={"source": "file2.txt"}),
        Document(page_content="World news and updates", metadata={"source": "file3.txt"}),
    ]


def test_build_index_creates_directory(tmp_path):
    index_dir = tmp_path / "whoosh_idx"
    docs = make_docs()

    # Directory does not exist before
    assert not index_dir.exists()

    build_whoosh_index(index_dir=str(index_dir), docs=docs)

    # Index directory should now exist and contain files
    assert index_dir.exists()
    contents = list(index_dir.iterdir())
    assert contents, "Index directory should not be empty after build"


def test_index_searchable(tmp_path):
    index_dir = tmp_path / "whoosh_idx"
    docs = make_docs()
    build_whoosh_index(index_dir=str(index_dir), docs=docs)

    # Open index and prepare parser
    ix = whoosh_index.open_dir(str(index_dir))
    searcher = ix.searcher()
    parser = QueryParser("content", schema=ix.schema)

    # Search for term present in first document
    q = parser.parse("World")
    hits = searcher.search(q, limit=5)
    assert len(hits) == 2 
    # confirm file1.txt is returned
    found_sources = {hit['path'] for hit in hits}
    assert "file1.txt" in found_sources

    # Search for term in second doc
    q2 = parser.parse("second")
    hits2 = searcher.search(q2, limit=5)
    assert len(hits2) == 1
    assert hits2[0]['path'] == "file2.txt"

    searcher.close()


def test_rebuild_clears_old_index(tmp_path):
    index_dir = tmp_path / "whoosh_idx"
    docs1 = make_docs()

    # Initial build
    build_whoosh_index(index_dir=str(index_dir), docs=docs1)
    # Create a dummy file in index_dir
    dummy = index_dir / "dummy.txt"
    dummy.write_text("trash")
    assert dummy.exists()

    # Build again with subset of docs
    docs2 = docs1[:1]
    build_whoosh_index(index_dir=str(index_dir), docs=docs2)

    # Dummy file should be removed
    assert not dummy.exists()
    # Index should reflect only docs2
    ix = whoosh_index.open_dir(str(index_dir))
    searcher = ix.searcher()
    parser = QueryParser("content", schema=ix.schema)
    # Search for content unique to docs1[1]
    q = parser.parse("second")
    hits = searcher.search(q, limit=5)
    assert len(hits) == 0
    searcher.close()


def test_build_index_with_empty_docs(tmp_path):
    index_dir = tmp_path / "whoosh_empty"
    docs = []  # empty list allowed

    # Should not error
    build_whoosh_index(index_dir=str(index_dir), docs=docs)
    assert index_dir.exists()
    # Search returns nothing
    ix = whoosh_index.open_dir(str(index_dir))
    searcher = ix.searcher()
    parser = QueryParser("content", schema=ix.schema)
    q = parser.parse("anything")
    hits = searcher.search(q, limit=5)
    assert len(hits) == 0
    searcher.close()

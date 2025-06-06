import re

import pytest
from langchain.schema import Document

from src.processing.preprocess import normalize_documents, chunk_documents

def test_normalize_documents_whitespace_and_nonprintables():
    # Create a document with mixed whitespace, Windows line endings, tabs, and non-printable chars
    raw_text = "  This is  a \t test.\r\nLine two.\rLine three.\x00\x1F End   "
    doc = Document(page_content=raw_text, metadata={"foo": "bar"})
    normalized = normalize_documents([doc])
    
    assert len(normalized) == 1
    norm = normalized[0].page_content
    
    # Expect single spaces, Unix line endings, no non-printables, trimmed ends
    assert "\r" not in norm
    assert "\n" in norm
    assert re.search(r"\s{2,}", norm) is None       # no double spaces
    assert "\t" not in norm
    assert "\x00" not in norm and "\x1F" not in norm
    assert norm.startswith("This is a test.")
    assert norm.endswith("End")

    # Metadata should be unchanged
    assert normalized[0].metadata == {"foo": "bar"}

def test_normalize_documents_multiple():
    # Two docs ensure list handling
    docs = [
        Document(page_content="A\tB", metadata={}),
        Document(page_content="   Leading and trailing   ", metadata={})
    ]
    out = normalize_documents(docs)
    assert len(out) == 2
    assert out[0].page_content == "A B"
    assert out[1].page_content == "Leading and trailing"


def test_chunk_documents_empty_input():
    """
    If no documents are provided, chunk_documents should return an empty list.
    """
    assert chunk_documents([], chunk_size=10, chunk_overlap=2) == []


def test_chunk_documents_shorter_than_chunk_size():
    """
    A document shorter than chunk_size should yield exactly one chunk identical to the original content,
    and preserve 'source' metadata with a 'chunk_id'.
    """
    text = "HelloWorld"
    metadata = {"source": "doc1.txt"}
    doc = Document(page_content=text, metadata=metadata)

    chunks = chunk_documents([doc], chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].page_content == text
    md = chunks[0].metadata
    assert md["source"] == "doc1.txt"
    assert "chunk_id" in md and md["chunk_id"].startswith("doc1.txt__chunk_")


def test_chunk_documents_exact_chunk_size():
    """
    A document whose length equals chunk_size should yield one chunk,
    preserve 'source' metadata and add 'chunk_id'.
    """
    text = "X" * 20
    metadata = {"source": "exact.txt"}
    doc = Document(page_content=text, metadata=metadata)

    chunks = chunk_documents([doc], chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].page_content == text
    md = chunks[0].metadata
    assert md["source"] == "exact.txt"
    assert "chunk_id" in md and md["chunk_id"].startswith("exact.txt__chunk_")


def test_chunk_documents_with_overlap():
    """
    Splitting a longer document should produce overlapping chunks at correct offsets,
    each preserving 'source' metadata and having correct 'chunk_id'.
    """
    text = ''.join(str(i % 10) for i in range(50))
    metadata = {"source": "file1.txt"}
    doc = Document(page_content=text, metadata=metadata)

    chunks = chunk_documents([doc], chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 3
    assert chunks[0].page_content == text[0:20]
    assert chunks[1].page_content == text[15:35]
    assert chunks[2].page_content == text[30:50]

    for idx, c in enumerate(chunks):
        md = c.metadata
        assert md["source"] == "file1.txt"
        assert md["chunk_id"] == f"file1.txt__chunk_{idx}"


def test_chunk_documents_multiple_documents():
    """
    Verify that multiple input documents are chunked independently,
    each preserving 'source' and unique 'chunk_id'.
    """
    doc1 = Document(page_content="A" * 30, metadata={"source": "doc1.txt"})
    doc2 = Document(page_content="B" * 30, metadata={"source": "doc2.txt"})

    chunks = chunk_documents([doc1, doc2], chunk_size=10, chunk_overlap=2)
    assert len(chunks) == 8

    for idx, chunk in enumerate(chunks[:4]):
        md = chunk.metadata
        assert md["source"] == "doc1.txt"
        assert md["chunk_id"] == f"doc1.txt__chunk_{idx}"

    for idx, chunk in enumerate(chunks[4:]):
        md = chunk.metadata
        assert md["source"] == "doc2.txt"
        assert md["chunk_id"] == f"doc2.txt__chunk_{idx}"

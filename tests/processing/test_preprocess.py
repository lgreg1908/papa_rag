import re

import pytest
from langchain.schema import Document

from src.processing.preprocess import normalize_documents

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

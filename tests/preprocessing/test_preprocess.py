import os
import re
from pathlib import Path
from datetime import datetime

import pytest
from langchain.schema import Document

from src.preprocessing.preprocess import normalize_documents, extract_metadata

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

def test_extract_metadata_with_existing_file(tmp_path: Path):
    # Create a temp file to serve as source
    file = tmp_path / "sample.txt"
    content = "hello"
    file.write_text(content, encoding="utf-8")
    # Record its creation time for test stability
    # (on some filesystems ctime may round, so allow a small delta)
    ctime = os.path.getctime(str(file))
    
    # Build Document pointing to that file
    doc = Document(page_content=content, metadata={"source": str(file)})
    metas = extract_metadata([doc])
    
    assert len(metas) == 1
    meta = metas[0]
    # Should have added char_count and created_at
    assert meta["char_count"] == len(content)
    # created_at parsed from ctime
    dt = datetime.fromisoformat(meta["created_at"])
    # within a second
    assert abs((dt - datetime.fromtimestamp(ctime)).total_seconds()) < 1.0

def test_extract_metadata_without_source_or_nonexistent(tmp_path: Path):
    # Doc with no source key
    doc1 = Document(page_content="x", metadata={})
    # Doc with nonexistent source
    doc2 = Document(page_content="y", metadata={"source": str(tmp_path/"nope.txt")})
    metas = extract_metadata([doc1, doc2])
    
    # Neither should get char_count or created_at
    assert all("char_count" not in m and "created_at" not in m for m in metas)

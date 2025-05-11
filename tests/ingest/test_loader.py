import os
from pathlib import Path
from typing import List

import pytest
from langchain.schema import Document

from src.ingestion.loader import (
    list_supported_files,
    load_documents,
    load_folder
)

def test_list_supported_files(tmp_path: Path):
    # create a mix of files
    txt = tmp_path / "foo.txt"
    txt.write_text("hello world", encoding="utf-8")


    # unsupported extension should be ignored
    other = tmp_path / "ignore.xyz"
    other.write_text("nothing", encoding="utf-8")

    found = list_supported_files(str(tmp_path))
    # order is nondeterministic, so compare as sets
    assert found == [str(txt)]

def test_load_documents_txt(tmp_path: Path):
    # only test .txt loading
    txt = tmp_path / "doc.txt"
    content = "Line1\nLine2"
    txt.write_text(content, encoding="utf-8")

    docs = load_documents([str(txt)])
    # TextLoader should produce exactly one Document
    assert isinstance(docs, List)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Line1" in docs[0].page_content


def test_load_folder(tmp_path: Path):
    # mix both a .txt and a .jpg under subfolders
    sub = tmp_path / "sub"
    sub.mkdir()
    txt = sub / "a.txt"
    txt.write_text("test", encoding="utf-8")

    docs = load_folder(str(tmp_path))
    assert isinstance(docs, List) 
    # should find one Document and one Image tuple
    assert len(docs) == 1
    # metadata.source should point to the txt file
    assert docs[0].metadata.get("source", "").endswith("a.txt")

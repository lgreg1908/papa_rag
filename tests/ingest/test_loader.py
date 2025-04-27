import os
from pathlib import Path
from typing import List, Tuple

import pytest
from PIL import Image
from langchain.schema import Document

from src.ingestion.loader import (
    list_supported_files,
    load_documents,
    load_images,
    load_folder
)

def test_list_supported_files(tmp_path: Path):
    # create a mix of files
    txt = tmp_path / "foo.txt"
    txt.write_text("hello world", encoding="utf-8")

    jpg = tmp_path / "pic.jpg"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(jpg)

    # unsupported extension should be ignored
    other = tmp_path / "ignore.xyz"
    other.write_text("nothing", encoding="utf-8")

    found = list_supported_files(str(tmp_path))
    # order is nondeterministic, so compare as sets
    assert set(found) == {str(txt), str(jpg)}

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

def test_load_images(tmp_path: Path):
    # create and load a PNG
    png = tmp_path / "img.png"
    img = Image.new("RGB", (5, 5), color="blue")
    img.save(png)

    loaded = load_images([str(png)])
    assert isinstance(loaded, List)
    assert len(loaded) == 1
    path, pil_img = loaded[0]
    assert path == str(png)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (5, 5)

def test_load_folder(tmp_path: Path):
    # mix both a .txt and a .jpg under subfolders
    sub = tmp_path / "sub"
    sub.mkdir()
    txt = sub / "a.txt"
    txt.write_text("test", encoding="utf-8")
    jpg = sub / "b.jpg"
    Image.new("RGB", (2,2)).save(jpg)

    docs, imgs = load_folder(str(tmp_path))
    assert isinstance(docs, List) and isinstance(imgs, List)
    # should find one Document and one Image tuple
    assert len(docs) == 1
    assert len(imgs) == 1
    # metadata.source should point to the txt file
    assert docs[0].metadata.get("source", "").endswith("a.txt")
    assert imgs[0][0].endswith("b.jpg")
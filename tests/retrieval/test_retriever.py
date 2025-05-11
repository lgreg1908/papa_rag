import pytest
from typing import List
from langchain.schema import Document

import src.retrieval.retriever as retriever_mod
from src.retrieval.retriever import Retriever

class DummyVectorStore:
    def __init__(self, to_return: List[Document]):
        self.to_return = to_return
        self.last_query = None
    def search(self, query_embedding: List[float], top_k: int) -> List[Document]:
        # record the passed query embedding and top_k
        self.last_query = (tuple(query_embedding), top_k)
        return list(self.to_return)

class DummyHit:
    def __init__(self, path: str, score: float):
        self._path = path
        self.score = score
    def get(self, key: str):
        if key == 'path':
            return self._path
        return None

class DummySearcher:
    def __init__(self, hits: List[DummyHit]):
        self.hits = hits
        self.last_qs = None
    def search(self, qs, limit=None):
        self.last_qs = qs
        return self.hits
    def close(self):
        pass

class DummyParser:
    def __init__(self):
        self.last_text = None
    def parse(self, text: str):
        self.last_text = text
        return f"PARSED({text})"

@pytest.fixture(autouse=True)
def patch_get_text_embeddings(monkeypatch):
    # Patch get_text_embeddings to return a fixed embedding
    def fake_get_text_embeddings(texts: List[str]):
        # return a 3-dimensional dummy vector per input
        return [[1.0, 2.0, 3.0] for _ in texts]
    monkeypatch.setattr(retriever_mod, 'get_text_embeddings', fake_get_text_embeddings)
    yield

def test_retrieve_returns_vector_store_results_only():
    # Prepare dummy documents
    docs = [Document(page_content='', metadata={'source': 'doc1'})]
    store = DummyVectorStore(to_return=docs)
    retr = Retriever(vector_store=store)

    results = retr.retrieve('hello', top_k=5, fallback=False)
    # Should return exactly the dummy docs
    assert results == docs
    # Ensure vector_store.search was called with our fake embedding
    expected_vec = (1.0, 2.0, 3.0)
    assert store.last_query == (expected_vec, 5)

def test_retrieve_with_fallback(monkeypatch):
    # Empty vector store results
    store = DummyVectorStore(to_return=[])
    retr = Retriever(vector_store=store)
    # Attach dummy Whoosh searcher and parser
    hits = [DummyHit('pathA', 0.9), DummyHit('pathB', 0.8)]
    searcher = DummySearcher(hits)
    parser = DummyParser()
    retr.whoosh_searcher = searcher
    retr.whoosh_parser = parser

    # Perform retrieve with fallback
    results = retr.retrieve('test query', top_k=2, fallback=True)
    # Expect two fallback hits as Document objects
    assert len(results) == 2
    assert results[0].metadata['source'] == 'pathA'
    assert results[0].metadata['score'] == 0.9
    assert results[1].metadata['source'] == 'pathB'
    assert results[1].metadata['score'] == 0.8
    # Parser should have recorded the query text
    assert parser.last_text == 'test query'
    # Searcher should have received the parsed query
    assert searcher.last_qs == 'PARSED(test query)'

def test_retrieve_no_fallback_when_disabled(monkeypatch):
    docs = [Document(page_content='', metadata={'source': 'only'})]
    store = DummyVectorStore(to_return=docs)
    retr = Retriever(vector_store=store)
    # even if we attach whoosh, fallback=False should ignore
    retr.whoosh_searcher = DummySearcher([DummyHit('x', 1.0)])
    retr.whoosh_parser = DummyParser()

    results = retr.retrieve('ignore fallback', top_k=3, fallback=False)
    assert results == docs

def test_retrieve_raises_on_non_string_query():
    store = DummyVectorStore(to_return=[])
    retr = Retriever(vector_store=store)
    with pytest.raises(ValueError):
        retr.retrieve(query=123, top_k=1)

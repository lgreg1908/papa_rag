import pytest
from typing import List

from src.processing.embeddings import get_text_embeddings, txt_cache

# Dummy classes for text embedding tests
class DummyData:
    def __init__(self, embedding):
        self.embedding = embedding

class DummyResponse:
    def __init__(self, data):
        self.data = data

class DummyTextClient:
    def __init__(self):
        self.embeddings = self
        self.calls = []

    def create(self, input: List[str], model: str):
        self.calls.append((tuple(input), model))
        return DummyResponse([DummyData([float(len(s))]) for s in input])

@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    # Patch OpenAI client for text embeddings
    from src.processing import embeddings as emb_mod
    dummy_text = DummyTextClient()
    monkeypatch.setattr(emb_mod, 'client', dummy_text)
    # Clear text cache before each test
    txt_cache.clear()
    return dummy_text


def test_get_text_embeddings_basic(patch_client):
    inputs = ["a", "bb", "ccc"]
    embs = get_text_embeddings(inputs)
    # Should call API once
    assert len(patch_client.calls) == 1
    assert patch_client.calls[0][0] == tuple(inputs)
    # Embeddings should match dummy logic (string lengths)
    assert embs == [[1.0], [2.0], [3.0]]


def test_get_text_embeddings_caching(patch_client):
    inputs = ["x", "x", "y"]
    embs1 = get_text_embeddings(inputs)
    assert len(patch_client.calls) == 1
    # Clear recorded calls, keep cache
    patch_client.calls.clear()
    embs2 = get_text_embeddings(inputs)
    # No new API calls on cached inputs
    assert patch_client.calls == []
    assert embs2 == embs1

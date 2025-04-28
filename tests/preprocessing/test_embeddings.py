import os
import pytest
from typing import List
from openai import OpenAI
from src.preprocessing.embeddings import get_text_embeddings, txt_cache

class DummyData:
    def __init__(self, embedding):
        self.embedding = embedding

class DummyResponse:
    def __init__(self, data):
        self.data = data

class DummyClient:
    def __init__(self):
        self.embeddings = self
        self.calls = []

    def create(self, input: List[str], model: str):
        # record that we were called
        self.calls.append((tuple(input), model))
        # return a DummyResponse: echo back simple vectors
        return DummyResponse([DummyData([float(len(s))]) for s in input])

@pytest.fixture(autouse=True)
def patch_openai_client(monkeypatch):
    # Replace the OpenAI client in our module with DummyClient
    from src.preprocessing.embeddings import client as real_client
    dummy = DummyClient()
    monkeypatch.setattr("src.preprocessing.embeddings.client", dummy)
    # Clear cache before each test
    txt_cache.clear()
    return dummy

def test_get_text_embeddings_basic(patch_openai_client):
    inputs = ["a", "bb", "ccc"]
    embs = get_text_embeddings(inputs)

    # DummyClient should have been called once
    assert len(patch_openai_client.calls) == 1
    called_input, called_model = patch_openai_client.calls[0]
    assert called_input == tuple(inputs)
    assert isinstance(called_model, str)

    # Embeddings should be lists of floats equal to string lengths
    assert embs == [[1.0], [2.0], [3.0]]

def test_get_text_embeddings_caching(patch_openai_client):
    inputs = ["repeat", "repeat", "unique"]
    embs1 = get_text_embeddings(inputs)
    # First call: one batch, API called once for both distinct strings
    assert len(patch_openai_client.calls) == 1

    # Clear only the recorded calls, but keep cache
    patch_openai_client.calls.clear()

    # Call again: both “repeat” and “unique” are now cached
    embs2 = get_text_embeddings(inputs)
    # No new API calls should have been made
    assert patch_openai_client.calls == []

    # Results identical
    assert embs2 == embs1

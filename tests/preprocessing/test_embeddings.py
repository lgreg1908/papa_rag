import pytest
import numpy as np
from PIL import Image
from typing import List

from src.preprocessing.embeddings import (
    get_text_embeddings,
    txt_cache,
    get_image_embeddings,
    img_cache,
)

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

# Dummy class for image embedding tests
class DummyImageModel:
    def __init__(self, model_name: str):
        pass

    def encode(self, images: List[Image.Image], batch_size: int, convert_to_numpy: bool):
        # Return numpy array where each vector equals its position index
        vec = np.arange(len(images), dtype=float).reshape(-1, 1)
        return vec

@pytest.fixture(autouse=True)
def patch_clients(monkeypatch):
    # Patch clients for embeddings
    from src.preprocessing import embeddings as emb_mod
    dummy_text = DummyTextClient()
    monkeypatch.setattr(emb_mod, 'client', dummy_text)
    monkeypatch.setattr(emb_mod, 'SentenceTransformer', DummyImageModel)
    # Clear caches before each test
    txt_cache.clear()
    img_cache.clear()
    return dummy_text


def test_get_text_embeddings_basic(patch_clients):
    inputs = ["a", "bb", "ccc"]
    embs = get_text_embeddings(inputs)
    # Should call API once
    assert len(patch_clients.calls) == 1
    assert patch_clients.calls[0][0] == tuple(inputs)
    # Embeddings should match dummy logic (string lengths)
    assert embs == [[1.0], [2.0], [3.0]]


def test_get_text_embeddings_caching(patch_clients):
    inputs = ["x", "x", "y"]
    embs1 = get_text_embeddings(inputs)
    assert len(patch_clients.calls) == 1
    patch_clients.calls.clear()
    embs2 = get_text_embeddings(inputs)
    # No new API calls on cached inputs
    assert patch_clients.calls == []
    assert embs2 == embs1


def test_get_image_embeddings_basic(patch_clients):
    img1 = Image.new('RGB', (2, 2), color='red')
    img2 = Image.new('RGB', (3, 3), color='blue')
    embs = get_image_embeddings([img1, img2])
    # Two distinct images => two vectors [0.0], [1.0]
    assert isinstance(embs, list)
    assert embs == [[0.0], [1.0]]
    # Cache should have entries for both images
    assert len(img_cache) == 2


def test_get_image_embeddings_caching(patch_clients):
    img = Image.new('RGB', (4, 4), color='green')
    # First call populates cache with one key
    embs1 = get_image_embeddings([img, img])
    assert len(img_cache) == 1
    cached_vec = next(iter(img_cache.values()))
    # Second call should return the same cached vector twice
    embs2 = get_image_embeddings([img, img])
    assert embs2 == [cached_vec, cached_vec]
    assert len(img_cache) == 1

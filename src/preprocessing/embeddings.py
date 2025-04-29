import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from PIL import Image

# Load env vars
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Model and batching parameters
TEXT_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
TEXT_BATCH_SIZE = 16
CLIP_MODEL = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")


# Simple in-memory cache
txt_cache: Dict[str, List[float]] = {}
img_cache: Dict[bytes, List[float]] = {}

def get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings for a list of texts using the OpenAI v1 client, with in-memory caching.

    Args:
        texts: List of input strings.

    Returns:
        A list of embedding vectors (one per input string).
    """
    # Prepare output list and collect uncached inputs
    embeddings: List[List[float]] = [None] * len(texts)
    to_request: List[str] = []
    idx_map: List[int] = []

    for idx, txt in enumerate(texts):
        if txt in txt_cache:
            embeddings[idx] = txt_cache[txt]
        else:
            idx_map.append(idx)
            to_request.append(txt)

    # Send in batches to avoid rate limits
    for i in range(0, len(to_request), TEXT_BATCH_SIZE):
        batch = to_request[i : i + TEXT_BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=TEXT_EMBED_MODEL)
        for j, item in enumerate(response.data):
            vector = item.embedding
            orig_idx = idx_map[i + j]
            embeddings[orig_idx] = vector
            txt_cache[batch[j]] = vector

    return embeddings


def get_image_embeddings(images: List[Image.Image]) -> List[List[float]]:
    """
    Returns embeddings for a list of PIL Images using a CLIP model, with caching on image bytes.

    Args:
        images: List of PIL Image objects.

    Returns:
        A list of embedding vectors (one per image).
    """
    # Initialize model once
    model = SentenceTransformer(CLIP_MODEL)

    outputs: List[List[float]] = [None] * len(images)
    uncached: List[Image.Image] = []
    idx_map: List[int] = []

    # Identify uncached images
    for idx, img in enumerate(images):
        key = img.tobytes()
        if key in img_cache:
            outputs[idx] = img_cache[key]
        else:
            idx_map.append(idx)
            uncached.append(img)

    # Compute embeddings for uncached images
    if uncached:
        vecs = model.encode(uncached, batch_size=len(uncached), convert_to_numpy=True)
        for j, vec in enumerate(vecs):
            orig_idx = idx_map[j]
            vector = vec.tolist()
            outputs[orig_idx] = vector
            img_cache[uncached[j].tobytes()] = vector

    return outputs


def main() -> None:
    """
    Demonstration of text and image embeddings.
    """
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Streamlit makes it easy to build data apps in Python."
    ]
    print(f"Text inputs ({len(texts)}):")
    for i, t in enumerate(texts, start=1):
        print(f"  {i}. {t}")

    text_embs = get_text_embeddings(texts)
    print("\nText Embeddings (first 5 dims):")
    for i, vec in enumerate(text_embs, start=1):
        print(f"  Sample {i}: {vec[:5]}... (len={len(vec)})")

    # Sample images: create two 10x10 red and blue squares
    img1 = Image.new("RGB", (10, 10), color="red")
    img2 = Image.new("RGB", (10, 10), color="blue")
    print(f"\nImage inputs: 2 generated images")

    img_embs = get_image_embeddings([img1, img2])
    print("Image Embeddings (first 5 dims):")
    for i, vec in enumerate(img_embs, start=1):
        print(f"  Image {i}: {vec[:5]}... (len={len(vec)})")

if __name__ == "__main__":
    main()


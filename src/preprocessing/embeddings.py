import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model and batching parameters
TEXT_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
TEXT_BATCH_SIZE = 16

# Simple in-memory cache
txt_cache: Dict[str, List[float]] = {}

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


def main() -> None:
    """
    Demo of get_text_embeddings:
      - Defines a few sample sentences
      - Calls get_text_embeddings on them
      - Prints out the resulting vectors (truncated)
    """
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Streamlit makes it easy to build data apps in Python.",
        "OpenAI embeddings capture semantic meaning."
    ]
    print(f"Input texts ({len(samples)}):")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. {s}")

    # Compute embeddings
    embs = get_text_embeddings(samples)

    print("\nEmbeddings:")
    for i, vec in enumerate(embs, 1):
        # Show only first 5 dimensions for brevity
        print(f"  Sample {i}: [{', '.join(f'{v:.4f}' for v in vec[:5])} ...] (len={len(vec)})")


if __name__ == "__main__":
    main()


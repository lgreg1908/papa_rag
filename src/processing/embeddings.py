import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from PIL import Image
import shutil

# Load env vars
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

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


def embed_documents(docs: List[Document]) -> List[Document]:
    """
    Adds text embeddings to each Document's metadata under 'embedding'.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        A new list of Document objects with embeddings in metadata.
    """
    texts = [d.page_content for d in docs]
    embs = get_text_embeddings(texts)
    result: List[Document] = []
    for doc, vec in zip(docs, embs):
        meta = dict(doc.metadata)
        meta['embedding'] = vec
        result.append(Document(page_content=doc.page_content, metadata=meta))
    return result

def main() -> None:
    """
    Demonstration of embed_documents.
    """
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Streamlit makes it easy to build data apps in Python."
    ]
    docs = [Document(page_content=t, metadata={}) for t in texts]
    print(f"Embedding {len(docs)} sample documents...")
    embedded_docs = embed_documents(docs)
    for i, doc in enumerate(embedded_docs, start=1):
        vec = doc.metadata['embedding']
        print(f"Doc {i}: first 5 dims: {vec[:5]}... length {len(vec)}")

if __name__ == "__main__":
    main()


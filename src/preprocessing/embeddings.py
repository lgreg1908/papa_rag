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


def embed_images(paths: List[str]) -> Dict[str, List[float]]:
    """
    Returns a mapping from image file path to its embedding vector.

    Args:
        paths: List of image file paths.

    Returns:
        Dict mapping path to embedding vector.
    """
    images = [Image.open(p) for p in paths]
    vecs = get_image_embeddings(images)
    return {p: v for p, v in zip(paths, vecs)}
def main() -> None:
    """
    Demonstration of embed_documents and embed_images.
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

    # Sample images saved to temp
    temp_dir = os.path.join(os.getcwd(), 'temp_emb_imgs')
    os.makedirs(temp_dir, exist_ok=True)
    paths: List[str] = []
    for color, fname in [('red', 'red.png'), ('blue', 'blue.png')]:
        p = os.path.join(temp_dir, fname)
        Image.new("RGB", (10, 10), color=color).save(p)
        paths.append(p)
    print(f"\nEmbedding {len(paths)} sample images... stored at: {paths}")
    embedded_imgs = embed_images(paths)
    for i, p in enumerate(paths, start=1):
        vec = embedded_imgs[p]
        print(f"Img {i} ({os.path.basename(p)}): first 5 dims: {vec[:5]}... length {len(vec)}")
    try:
        shutil.rmtree(temp_dir)
        print(f"Removed temp directory: {temp_dir}")
    except Exception as e:
        print(f"Failed to remove temp dir {temp_dir}: {e}")

if __name__ == "__main__":
    main()


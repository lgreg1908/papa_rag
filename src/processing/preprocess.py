import re
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def normalize_documents(docs: List[Document]) -> List[Document]:
    """
    Normalize text content for each Document:
      - Strip extra whitespace
      - Normalize line endings
      - Remove non-printable characters

    Args:
        docs: List of LangChain Document objects
    Returns:
        List of new Document objects with cleaned text and same metadata
    """
    normalized = []
    for doc in docs:
        text = doc.page_content
        # Normalize whitespace and line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r"[ \t]+", " ", text)
        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E\n]", "", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        # Rebuild document
        normalized.append(
            Document(page_content=text, metadata=dict(doc.metadata))
        )
    return normalized

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks.

    Args:
        docs (List[Document]): List of pre-normalized LangChain Document objects.
        chunk_size (int): Maximum number of characters (or tokens) per chunk.
        chunk_overlap (int): Number of characters (or tokens) to overlap between chunks.

    Returns:
        List[Document]: A flat list of Document objects, each containing
                        a chunk of the original text with preserved metadata.
    """
    splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs: List[Document] = splitter.split_documents(docs)
    return chunked_docs

def main() -> None:
    """
    Demonstration of normalize_documents and extract_metadata functions.
    """
    # Example: load sample docs
    from src.ingestion.loader import load_folder

    folder = 'data/sample'
    docs = load_folder(folder)
    print(f"Loaded {len(docs)} raw document chunks.")

    # Normalize
    norm_docs = normalize_documents(docs)
    print(f"Normalized {len(norm_docs)} documents.\nSample content:\n{norm_docs[0].page_content[:25]}...\n")

    # Chunk
    chunk_docs = chunk_documents(norm_docs)
    print(f"Created {len(chunk_docs)} chunks from {len(docs)} documents.")

if __name__ == '__main__':
    main()

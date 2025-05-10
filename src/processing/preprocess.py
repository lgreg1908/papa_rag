import re
from typing import List
from langchain.schema import Document


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
    print(f"Normalized {len(norm_docs)} documents.\nSample content:\n{norm_docs[0].page_content[:100]}...\n")

if __name__ == '__main__':
    main()

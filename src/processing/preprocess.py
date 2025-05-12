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
    Split documents into fixed-size, overlapping chunks—and tag each one with
    a unique `chunk_id` based on its source path and position.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("file_path", "")
        # split_documents returns a list of Documents
        for i, piece in enumerate(splitter.split_documents([doc])):
            meta = dict(doc.metadata)  # copy original metadata
            # create a unique chunk identifier
            meta["chunk_id"] = f"{src}__chunk_{i}"
            chunked.append(
                Document(page_content=piece.page_content, metadata=meta)
            )
    return chunked

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

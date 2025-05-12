import os
from src.utils.logger import logger
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader

# Supported extensions for text documents and images
TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def list_supported_files(folder_path: str) -> List[str]:
    """
    Recursively search for and list all supported document files in a directory.

    This function walks through the directory tree rooted at `folder_path`,
    and collects paths for files whose extensions match those defined in TEXT_EXTENSIONS.

    Args:
        folder_path (str): The root directory to begin the search.

    Returns:
        List[str]: A list of full file paths for supported document types found under the directory.
    """
    file_paths: List[str] = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in TEXT_EXTENSIONS:
                file_paths.append(os.path.join(root, fname))
    return file_paths

def load_documents(paths: List[str]) -> List[Document]:
    """
    Load a batch of files into LangChain Document objects, using the appropriate loader
    for each supported extension. Errors during loading are logged as warnings and
    processing continues with the next file.

    Args:
        paths (List[str]): A list of file paths to load. Supported extensions are:
            - .pdf   → PyPDFLoader
            - .docx  → Docx2txtLoader
            - .txt   → TextLoader (UTF-8)
            - .md    → UnstructuredMarkdownLoader

    Returns:
        List[Document]: A flat list of all Document objects produced by the loaders
                        for the successfully processed files.
    """
    docs: List[Document] = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
                docs.extend(loader.load())
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf8")
                docs.extend(loader.load())
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(path)
                docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"Error loading document {path}: {e}")
    return docs

def load_folder(folder_path: str) -> List[Document]:
    """
    Discover and load all supported documents in a directory.

    This function combines `list_supported_files` and `load_documents` to provide
    a one-stop method for retrieving all documents from a folder tree. Only files
    with extensions defined in TEXT_EXTENSIONS are considered.

    Args:
        folder_path (str): The directory path to scan and load documents from.

    Returns:
        List[Document]: A list of loaded Document objects ready for processing.

    """
    paths = list_supported_files(folder_path)
    text_paths = [i for i in paths if os.path.splitext(i)[1].lower()]
    docs = load_documents(text_paths)
    return docs

def main() -> None:
    folder_path = 'data/sample'
    print(f"Parsing the folder: {folder_path}")

    paths = list_supported_files(folder_path)
    print(f"Found {len(paths)} valid files.")

    docs = load_folder(folder_path=folder_path)
    print(f"Found {len(docs)} chunked docs.")
    print(f"Sample doc: {docs[1]}")


if __name__ == '__main__':
    main()





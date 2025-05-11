import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader

# Supported extensions for text documents and images
TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def list_supported_files(folder_path: str) -> List[str]:
    """
    Recursively list all supported files in the given folder.
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
    Load text documents from given file paths using LangChain loaders.
    Returns a list of LangChain Document objects.
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
            print(f"Error loading document {path}: {e}")
    return docs

def load_folder(folder_path: str) -> List[Document]:
    """
    Load all supported files in the folder:
      - Text documents → LangChain Document objects
      - Images → (path, PIL Image)

    Returns a tuple (documents, images).
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





import os
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
from PIL import Image

# Supported extensions for text documents and images
TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_supported_files(folder_path: str) -> List[str]:
    """
    Recursively list all supported files in the given folder.
    """
    file_paths: List[str] = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in TEXT_EXTENSIONS or ext in IMAGE_EXTENSIONS:
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

def load_images(paths: List[str]) -> List[Tuple[str, Image.Image]]:
    """
    Load image files from given paths using PIL.
    Returns a list of tuples: (file_path, PIL Image).
    """
    images: List[Tuple[str, Image.Image]] = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            try:
                img = Image.open(path)
                images.append((path, img))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
    return images


if __name__ == '__main__':

    files = list_supported_files('data/tmp')
    docs = load_documents(paths=files)
    print(len(docs))

    # print(files)




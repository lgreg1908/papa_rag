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

if __name__ == '__main__':
    os.makedirs(name='tmp',exist_ok=True)
    with open('tmp/test1.txt', 'w') as f:
        f.write('.')
    with open('tmp/test2.txt', 'w') as f:
        f.write('.')
    with open('tmp/test3.docx', 'w') as f:
        f.write('.')

    files = list_supported_files('tmp')
    print(files)
    import shutil
    shutil.rmtree('tmp')



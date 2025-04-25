
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable, List, Tuple
from langchain.schema import Document
from PIL import Image

from .loader import (
    TEXT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    load_documents,
    load_images,
)
from src.utils.logger import logger

class IngestionHandler(FileSystemEventHandler):
    """
    Handles filesystem events and triggers ingestion for added or modified files.
    """

    def __init__(
        self,
        ingest_callback: Callable[
            [List[Document], List[Tuple[str, Image.Image]]], None
        ],
    ):
        """
        Initialize the ingestion handler.

        Args:
            ingest_callback: A callback function that receives lists of Documents and image tuples.
        """
        super().__init__()
        self.ingest_callback = ingest_callback

    def on_created(self, event):
        """
        Called when a file or directory is created.

        Args:
            event: The filesystem event containing src_path and is_directory attributes.
        """
        if not event.is_directory:
            self._process(event.src_path)

    def on_modified(self, event):
        """
        Called when a file or directory is modified.

        Args:
            event: The filesystem event containing src_path and is_directory attributes.
        """
        if not event.is_directory:
            self._process(event.src_path)

    def _process(self, path: str):
        """
        Internal processing of a single file path.

        Determines file extension and loads documents or images accordingly,
        then invokes the ingest callback.

        Args:
            path: The path to the file to process.
        """
        ext = os.path.splitext(path)[1].lower()
        docs: List[Document] = []
        imgs: List[Tuple[str, Image.Image]] = []
        try:
            if ext in TEXT_EXTENSIONS:
                docs = load_documents([path])
                logger.info(f"Loaded {len(docs)} document chunks from {path}")
            elif ext in IMAGE_EXTENSIONS:
                imgs = load_images([path])
                logger.info(f"Loaded {len(imgs)} images from {path}")
            else:
                return

            # Trigger downstream ingestion
            self.ingest_callback(docs, imgs)
            logger.info(f"Ingestion callback executed for {path}")
        except Exception as e:
            logger.error(f"Error processing file {path}: {e}")
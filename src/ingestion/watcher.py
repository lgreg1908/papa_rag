
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from typing import Callable, List
from langchain.schema import Document

from src.ingestion.loader import (
    TEXT_EXTENSIONS,
    load_documents,
    load_folder
)
from src.utils.logger import logger


class IngestionHandler(FileSystemEventHandler):
    """
    Handles filesystem events and triggers ingestion for added or modified files.
    """

    def __init__(
        self,
        ingest_callback: Callable[
            [List[Document]], None
        ],
    ):
        """
        Initialize the ingestion handler.

        Args:
            ingest_callback: A callback function that receives lists of Documents and image tuples.
        """
        super().__init__()
        self.ingest_callback = ingest_callback

    def on_created(self, event: FileSystemEvent) -> None:
        """
        Called when a file or directory is created.

        Args:
            event: The filesystem event containing src_path and is_directory attributes.
        """
        if not event.is_directory:
            self._process(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """
        Called when a file or directory is modified.

        Args:
            event: The filesystem event containing src_path and is_directory attributes.
        """
        if not event.is_directory:
            self._process(event.src_path)

    def _process(self, path: str) -> None:
        """
        Internal processing of a single file path.

        Determines file extension and loads documents or images accordingly,
        then invokes the ingest callback.

        Args:
            path: The path to the file to process.
        """
        ext = os.path.splitext(path)[1].lower()
        docs: List[Document] = []
        try:
            if ext in TEXT_EXTENSIONS:
                docs = load_documents([path])
                logger.info(f"Loaded {len(docs)} document chunks from {path}")
            else:
                return

            # Trigger downstream ingestion
            self.ingest_callback(docs)
            logger.info(f"Ingestion callback executed for {path}")
        except Exception as e:
            logger.error(f"Error processing file {path}: {e}")


class FolderWatcher:
    """
    Watches a folder and ingests files on changes.
    """

    def __init__(
        self,
        folder_path: str,
        ingest_callback: Callable[
            [List[Document]], None
        ],
    ):
        """
        Initialize the folder watcher.

        Args:
            folder_path: Path of the folder to monitor.
            ingest_callback: Callback for ingestion of docs and images.
        """
        self.folder_path = folder_path
        self.ingest_callback = ingest_callback
        self.event_handler = IngestionHandler(self.ingest_callback)
        self.observer = Observer()

    def start(self) -> None:
        """
        Perform an initial full-folder ingest, then start monitoring for file changes.

        Ingests existing files, schedules the observer, and begins watching recursively.
        """
        try:
            docs = load_folder(self.folder_path)
            logger.info(
                f"Initial load: {len(docs)} docs from {self.folder_path}"
            )
            self.ingest_callback(docs)
        except Exception as e:
            logger.error(f"Error during initial folder load: {e}")

        self.observer.schedule(
            self.event_handler, self.folder_path, recursive=True
        )
        self.observer.start()
        logger.info(f"Started watching folder: {self.folder_path}")

    def stop(self) -> None:
        """
        Stop the observer and wait for its thread to terminate.
        """
        self.observer.stop()
        self.observer.join()
        logger.info(f"Stopped watching folder: {self.folder_path}")

    def run(self) -> None:
        """
        Convenience method to start the watcher and run indefinitely until interrupted.

        Listens for KeyboardInterrupt to stop gracefully.
        """
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

def main():
    """
    CLI entrypoint: watches 'data/tmp' folder and logs ingestion events to console.
    """
    folder = 'data/sample'
    def ingest_callback(docs: List[Document]):
        print(f"Ingested {len(docs)} documents from {folder}")

    watcher = FolderWatcher(folder, ingest_callback)
    watcher.run()


if __name__ == '__main__':
    main()

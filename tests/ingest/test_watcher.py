import time
from pathlib import Path
import threading
import pytest

from langchain.schema import Document
from src.ingestion.watcher import IngestionHandler, FolderWatcher

class DummyEvent:
    """Minimal FileSystemEvent stub."""
    def __init__(self, src_path: str, is_directory: bool = False):
        self.src_path = src_path
        self.is_directory = is_directory


def test_ingestion_handler_process_text(tmp_path: Path):
    # Create a sample .txt
    file = tmp_path / "sample.txt"
    file.write_text("hello world", encoding="utf-8")

    # Capture callback inputs
    received = []
    def cb(docs):
        received.append(docs)

    handler = IngestionHandler(cb)
    # Directly invoke the private _process
    handler._process(str(file))

    # Should have been called exactly once, docs non-empty
    assert len(received) == 1
    docs = received[0]
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)


def test_ingestion_handler_on_created_and_modified(tmp_path: Path):
    file = tmp_path / "foo.txt"
    file.write_text("data", encoding="utf-8")

    calls = []
    handler = IngestionHandler(lambda docs: calls.append(docs))

    # Simulate FileSystemEvent calls
    ev = DummyEvent(str(file), is_directory=False)
    handler.on_created(ev)
    handler.on_modified(ev)

    # Both should invoke _process
    assert len(calls) == 2
    for docs in calls:
        assert all(isinstance(d, Document) for d in docs)


def test_folder_watcher_initial_load(tmp_path: Path):
    # Prepare folder with one text file and one non-text file
    folder = tmp_path / "watch"
    folder.mkdir()

    txt = folder / "a.txt"
    txt.write_text("abc", encoding="utf-8")

    other = folder / "b.png"
    other.write_bytes(b"not an image")  # loader ignores non-text

    # Collect callback invocations
    calls = []
    def cb(docs):
        calls.append(docs)

    watcher = FolderWatcher(str(folder), cb)
    watcher.start()

    # Initial load only
    assert len(calls) == 1
    docs = calls[0]
    # Text loader yields at least one Document; non-text ignored
    assert any(isinstance(d, Document) for d in docs)

    watcher.stop()


@pytest.mark.timeout(2)
def test_folder_watcher_detects_new_file(tmp_path: Path):
    # Setup watcher on empty folder
    folder = tmp_path / "watch2"
    folder.mkdir()

    calls = []
    def cb(docs):
        calls.append(docs)

    watcher = FolderWatcher(str(folder), cb)
    # Run watcher in background
    thread = threading.Thread(target=watcher.run, daemon=True)
    thread.start()

    # Give it a moment to start
    time.sleep(0.2)

    # Create a new supported file
    new_file = folder / "new.txt"
    new_file.write_text("fresh", encoding="utf-8")

    # Wait for the event to be picked up
    time.sleep(0.5)

    watcher.stop()
    thread.join(timeout=1)

    # There should be at least two calls: initial load + the new file
    assert len(calls) >= 2
    # The last call should reflect the new file content
    docs = calls[-1]
    assert any("fresh" in d.page_content for d in docs)

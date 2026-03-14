import os
from pathlib import Path


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    return "integration" in path.parts and os.getenv("NANGILA_RUN_INTEGRATION") != "1"

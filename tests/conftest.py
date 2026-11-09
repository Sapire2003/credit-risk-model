"""Point the API at a throwaway SQLite database before any test imports it."""
import os
import tempfile
from pathlib import Path

os.environ["DATABASE_URL"] = f"sqlite:///{(Path(tempfile.mkdtemp()) / 'test.db').as_posix()}"
os.environ.pop("MODEL_S3_URI", None)

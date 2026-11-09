"""Find the trained model: download it from Amazon S3 when MODEL_S3_URI is set, else use the local file.

Keeping the model in S3 means the server never needs a copy baked into the code, and each version
is stored under its own key, so rolling back is just pointing MODEL_S3_URI at the previous file.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

LOCAL_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/path/to/file' -> ('bucket', 'path/to/file')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI needs a bucket and a key: {uri!r}")
    return bucket, key


def s3_client():
    import boto3  # imported here so local runs don't need AWS libraries loaded

    return boto3.client("s3")


def ensure_model(uri: str | None = None, cache_dir: Path | None = None, client=None) -> Path:
    """Return a local path to the model, downloading it from S3 once if needed."""
    uri = uri if uri is not None else os.environ.get("MODEL_S3_URI", "").strip()
    if not uri:
        return LOCAL_MODEL
    bucket, key = parse_s3_uri(uri)
    cache_dir = cache_dir or Path(os.environ.get("MODEL_CACHE_DIR", tempfile.gettempdir())) / "credit-model"
    target = cache_dir / key.replace("/", "__")
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    body = (client or s3_client()).get_object(Bucket=bucket, Key=key)["Body"].read()
    partial = target.with_suffix(".part")
    partial.write_bytes(body)
    partial.replace(target)  # rename last, so a failed download never leaves a broken model behind
    return target

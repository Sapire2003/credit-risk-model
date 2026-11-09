"""Upload the trained model to Amazon S3 under a versioned name.

Usage: python src/upload_model.py s3://your-bucket/models/
Prints the MODEL_S3_URI to give the server. Needs AWS credentials (run `aws configure` first).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_store import LOCAL_MODEL, parse_s3_uri, s3_client  # noqa: E402


def upload(prefix_uri: str, model_path: Path = LOCAL_MODEL, client=None) -> str:
    bucket, prefix = parse_s3_uri(prefix_uri.rstrip("/") + "/x")
    prefix = prefix[:-1]  # drop the placeholder, keeping the trailing slash
    version = joblib.load(model_path)["version"]
    key = f"{prefix}credit-default-{version}.joblib"
    (client or s3_client()).put_object(Bucket=bucket, Key=key, Body=model_path.read_bytes())
    return f"s3://{bucket}/{key}"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    uri = upload(sys.argv[1])
    print(f"Uploaded. Set this on the server:\nMODEL_S3_URI={uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download the public LendingClub loan file used by this project.

Source: H2O.ai's cleaned extract of LendingClub's public loan data
(163k resolved loans, one row per loan, features known at origination).
"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "loan.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, OUT)
    print(f"{OUT.name}: {OUT.stat().st_size:,} bytes, sha256={hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()

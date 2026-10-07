from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from src.eola_config import load_config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    config = load_config()
    root = Path(config["_project_root"])
    ds = config["dataset"]

    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    destination = raw_dir / ds["filename"]
    temporary = raw_dir / f"{ds['filename']}.part"

    expected_hash = ds["expected_sha256"]
    url = ds["source_url"]

    print(f"EOLA dataset version : {ds['version']}")
    print(f"Dataset DOI          : {ds['doi']}")
    print(f"Destination          : {destination}")

    if destination.exists():
        existing_hash = sha256(destination)

        if existing_hash == expected_hash:
            print(f"SHA-256              : {existing_hash}")
            print("DATASET ACQUISITION: ALREADY VALID")
            return

        raise RuntimeError(
            "A dataset already exists in data/raw but its SHA-256 "
            "does not match the canonical v1.0.4 release.\n"
            f"Observed: {existing_hash}\n"
            f"Expected: {expected_hash}\n"
            "The existing raw file was NOT modified."
        )

    if temporary.exists():
        temporary.unlink()

    print(f"Downloading from     : {url}")
    urllib.request.urlretrieve(url, temporary)

    downloaded_hash = sha256(temporary)

    print(f"Downloaded SHA-256   : {downloaded_hash}")

    if downloaded_hash != expected_hash:
        temporary.unlink(missing_ok=True)

        raise RuntimeError(
            "Downloaded file failed SHA-256 verification.\n"
            f"Observed: {downloaded_hash}\n"
            f"Expected: {expected_hash}"
        )

    shutil.move(str(temporary), str(destination))

    print("SHA-256              : PASS")
    print("DATASET ACQUISITION: PASS")


if __name__ == "__main__":
    main()
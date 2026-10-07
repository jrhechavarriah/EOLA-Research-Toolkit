from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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

    csv_path = root / "data" / "raw" / ds["filename"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {csv_path}\n"
            "Run: python src\\data_acquisition.py"
        )

    digest = sha256(csv_path)
    df = pd.read_csv(csv_path)

    observed_columns = df.columns.tolist()
    expected_columns = ds["expected_column_names"]

    target = config["modeling"]["target"]
    identifier = config["modeling"]["identifier"]

    checks = {
        "dataset_version": ds["version"],
        "dataset_doi": ds["doi"],
        "source_url": ds["source_url"],
        "path": csv_path.relative_to(root).as_posix(),
        "validated_utc": datetime.now(timezone.utc).isoformat(),

        "sha256": digest,
        "sha256_ok": digest == ds["expected_sha256"],

        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "shape_ok": df.shape == (
            ds["expected_rows"],
            ds["expected_columns"],
        ),

        "column_names": observed_columns,
        "schema_ok": observed_columns == expected_columns,

        "unique_PUBLIC_RECORD_ID": int(df[identifier].nunique()),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_cells": int(df.isna().sum().sum()),

        "target_counts": {
            str(k): int(v)
            for k, v in
            df[target].value_counts().sort_index().items()
        },
    }

    checks["contract_ok"] = all(
        [
            checks["sha256_ok"],
            checks["shape_ok"],
            checks["schema_ok"],
            checks["unique_PUBLIC_RECORD_ID"] == ds["expected_rows"],
            checks["duplicate_rows"] == 0,
            checks["null_cells"] == 0,
            checks["target_counts"] == {"0": 691, "1": 11941},
        ]
    )

    outputs = root / "outputs"
    metadata = root / "data" / "metadata"

    outputs.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    report = json.dumps(checks, indent=2, ensure_ascii=False)

    (outputs / "dataset_contract_validation.json").write_text(
        report,
        encoding="utf-8",
    )

    (metadata / "dataset_manifest.json").write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    if not checks["contract_ok"]:
        raise SystemExit("\nDATA CONTRACT: FAILED")

    print("\nDATA CONTRACT: PASS")


if __name__ == "__main__":
    main()
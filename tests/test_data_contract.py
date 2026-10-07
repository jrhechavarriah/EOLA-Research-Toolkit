from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "eola_config.yaml"


def load_config() -> dict:
    with CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_canonical_dataset_contract():
    config = load_config()
    ds = config["dataset"]

    csv_path = ROOT / "data" / "raw" / ds["filename"]

    assert csv_path.exists()
    assert sha256(csv_path) == ds["expected_sha256"]

    df = pd.read_csv(csv_path)

    assert df.shape == (
        ds["expected_rows"],
        ds["expected_columns"],
    )

    assert df.columns.tolist() == ds["expected_column_names"]

    identifier = config["modeling"]["identifier"]
    target = config["modeling"]["target"]

    assert df[identifier].nunique() == ds["expected_rows"]
    assert df.duplicated().sum() == 0
    assert df.isna().sum().sum() == 0

    assert (
        df[target]
        .value_counts()
        .sort_index()
        .to_dict()
        == {0: 691, 1: 11941}
    )
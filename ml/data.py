import json
from pathlib import Path

import pandas as pd

from data.data_prep import file_hash

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed-2025-v2"
FROZEN_MANIFEST = ROOT / "reports" / "split_audit" / "audit.json"


def verify_splits(folder=DATA_DIR, expected_path=FROZEN_MANIFEST):
    """Check every split against the committed frozen hashes."""
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    for name, spec in expected["splits"].items():
        path = Path(folder) / f"{name}.csv"
        if not path.is_file() or file_hash(path) != spec["sha256"]:
            raise ValueError(f"Frozen split hash mismatch: {name}. Restore the approved file; do not rebuild it.")
    return expected


def load_splits(folder=DATA_DIR, expected_path=FROZEN_MANIFEST):
    manifest = verify_splits(folder, expected_path)
    frames = {}
    for name, spec in manifest["splits"].items():
        df = pd.read_csv(Path(folder) / f"{name}.csv", dtype="string", keep_default_na=False)
        if len(df) != spec["rows"] or df["topic"].value_counts().to_dict() != spec["topics"]:
            raise ValueError(f"Frozen split counts changed: {name}")
        frames[name] = df
    return frames, manifest

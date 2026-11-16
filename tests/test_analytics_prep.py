import json
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.analytics_prep import prepare
from data.data_prep import file_hash


def test_analytics_preserves_rows_and_flags_cross_file_duplicates(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, ids, texts in [("a", [1, 2], ["Same text", ""]), ("b", [3, 4], [" same  TEXT ", "New text"])]:
        df = pd.DataFrame({"Complaint ID": ids, "Consumer complaint narrative": texts,
                           "Product": ["Card"] * 2, "Date received": ["2025-01-01"] * 2})
        with zipfile.ZipFile(raw / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}.csv", df.to_csv(index=False))
    output = tmp_path / "analytics"
    stats = prepare(raw, output)
    df = pd.concat(pd.read_parquet(path) for path in sorted(output.glob("*.parquet")))
    assert len(df) == stats["rows"] == 4
    assert stats["with_narrative"] == 3
    assert stats["duplicate_text"] == 1
    assert df["has_narrative"].tolist() == [True, False, True, True]
    assert df["is_duplicate_text"].tolist() == [False, False, True, False]
    assert df["text"].iloc[2] == " same  TEXT "
    assert json.loads((output / "manifest.json").read_text())["months"] == {"2025-01": 4}
    with pytest.raises(ValueError, match="empty"):
        prepare(raw, output)


def test_saved_analytics():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data" / "analytics-2025"
    if not (folder / "manifest.json").exists():
        pytest.skip("Run analytics preparation first")
    stats = json.loads((folder / "manifest.json").read_text())
    model = json.loads((root / "data" / "processed-2025" / "manifest.json").read_text())
    assert stats["rows"] == model["input_rows"] == sum(stats["months"].values())
    assert len(stats["months"]) == 12
    assert {(s["file"], s["sha256"]) for s in stats["sources"]} == {(s["file"], s["sha256"]) for s in model["sources"]}
    for source in stats["sources"]:
        path = folder / source["output"]
        assert pq.read_metadata(path).num_rows == source["rows"]
        assert file_hash(path) == source["output_sha256"]

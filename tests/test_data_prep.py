import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from data.data_prep import build_store, clean_text, file_hash, make_splits, sample_store


def test_clean_text_preserves_punctuation():
    assert clean_text("  Can't\n pay   $25!  ") == "Can't pay $25!"


def test_load_removes_empty_and_duplicate_rows(tmp_path):
    path = tmp_path / "complaints.csv"
    pd.DataFrame({
        "Complaint ID": [1, 2, 3, 4, 5, 6],
        "Consumer complaint narrative": ["Late  fee", "late fee", None, "  ", "Wrong bill", "Other"],
        "Product": ["Card", "Card", "Card", "Card", "Loan", None],
    }).to_csv(path, index=False)
    db = tmp_path / "cleaned.sqlite"
    build_store([path], db)
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query("SELECT * FROM complaints ORDER BY complaint_id", conn)
    assert df["complaint_id"].tolist() == ["1", "5"]
    assert df["text"].tolist() == ["Late fee", "Wrong bill"]


def test_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Product\nCard\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing columns"):
        build_store([path], tmp_path / "bad.sqlite")


def test_splits_are_repeatable_disjoint_and_stratified():
    df = pd.DataFrame({
        "complaint_id": range(1000),
        "text": [f"Complaint {i}" for i in range(1000)],
        "topic": ["Card"] * 600 + ["Loan"] * 380 + ["Small"] * 20,
    })
    splits = make_splits(df, 1000, min_class=30)
    again = make_splits(df, 1000, min_class=30)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [700, 150, 150]
    ids = []
    for name, split in splits.items():
        pd.testing.assert_frame_equal(split, again[name])
        assert set(split["topic"]) == {"Card", "Loan", "Other"}
        ids.extend(split["complaint_id"])
    assert len(set(ids)) == len(ids) == 1000
    assert "Small" in set(df["topic"])


def test_sample_keeps_requested_size():
    df = pd.DataFrame({"topic": ["A"] * 501 + ["B"] * 499})
    splits = make_splits(df, 333)
    assert sum(map(len, splits.values())) == 333


def test_rejects_insufficient_data():
    with pytest.raises(ValueError, match="only 2 remain"):
        make_splits(pd.DataFrame({"topic": ["A", "A"]}), 30)


def test_rejects_too_small_class():
    with pytest.raises(ValueError, match="at least 10"):
        make_splits(pd.DataFrame({"topic": ["A"] * 99 + ["B"]}), 100, min_class=1)


@pytest.mark.parametrize("name", ["processed", "processed-2025"])
def test_saved_real_data(name):
    folder = Path(__file__).resolve().parents[1] / "data" / name
    if not (folder / "manifest.json").exists():
        pytest.skip("Run data preparation on the CFPB export first")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if "drop_counts" in manifest:
        assert manifest["input_rows"] == manifest["clean_rows"] + sum(manifest["drop_counts"].values())
        assert sum(src["rows"] for src in manifest["sources"]) == manifest["input_rows"]
        with sqlite3.connect(folder / "cleaned.sqlite") as conn:
            assert conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0] == manifest["clean_rows"]
            if manifest["year"]:
                months = conn.execute("SELECT DISTINCT substr(date_received, 1, 7) FROM complaints").fetchall()
                assert {row[0] for row in months} == {f"{manifest['year']}-{m:02d}" for m in range(1, 13)}
    frames = []
    for name in ("train", "val", "test"):
        path = folder / f"{name}.csv"
        df = pd.read_csv(path, dtype="string", keep_default_na=False)
        saved = manifest["splits"][name]
        assert file_hash(path) == saved["sha256"]
        assert len(df) == saved["rows"]
        assert df["topic"].value_counts().to_dict() == saved["topics"]
        assert set(df["topic"]) == set(manifest["splits"]["train"]["topics"])
        assert df[["complaint_id", "text", "topic"]].ne("").all().all()
        frames.append(df)
    combined = pd.concat(frames)
    assert len(combined) == manifest["sample_size"]
    assert combined["complaint_id"].is_unique
    assert combined["text"].map(clean_text).str.casefold().is_unique
    for topic, count in combined["topic"].value_counts().items():
        for df, ratio in zip(frames, (0.70, 0.15, 0.15)):
            assert abs(df["topic"].eq(topic).sum() - count * ratio) <= 1


def test_chunked_drop_counts(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame({
        "Complaint ID": [None, "2", "3", "4", "5", "6", "6", "8", "8", "10"],
        "Consumer complaint narrative": ["x", " ", "x", "x", "x", "Alpha", "Beta", " ALPHA ", "Gamma", "Delta"],
        "Product": ["Card", "Card", None] + ["Card"] * 7,
        "Date received": ["2025-01-01"] * 3 + ["bad", "2024-12-31"] + ["2025-12-31"] * 5,
    }).to_csv(path, index=False)
    for chunksize in (2, 100):
        stats = build_store([path], tmp_path / f"{chunksize}.sqlite", year=2025, chunksize=chunksize)
        assert stats["input_rows"] == 10
        assert stats["kept_rows"] == 2
        assert stats["dropped"] == {
            "missing_id": 1, "missing_text": 1, "missing_topic": 1,
            "invalid_date": 1, "outside_year": 1, "duplicate_id": 2, "duplicate_text": 1,
        }


def test_store_sampling_across_archives(tmp_path):
    df = pd.DataFrame({
        "Complaint ID": range(200),
        "Consumer complaint narrative": [f"Complaint {i}" for i in range(200)],
        "Product": ["Card"] * 180 + ["Rare A"] * 10 + ["Rare B"] * 10,
    })
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    df.to_csv(first, index=False)
    df.iloc[-20:].to_csv(second, index=False)
    db = tmp_path / "cleaned.sqlite"
    stats = build_store([first, second], db, chunksize=30)
    assert stats["kept_rows"] == 200
    assert stats["dropped"]["duplicate_id"] == 20
    sample = sample_store(db, 100, min_class=15)
    assert sample["topic"].value_counts().to_dict() == {"Card": 90, "Other": 10}
    pd.testing.assert_frame_equal(sample, sample_store(db, 100, min_class=15))
    with pytest.raises(ValueError, match="already exists"):
        build_store([first], db)

import argparse
import hashlib
import json
import logging
import platform
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
SEED = 42
COLUMNS = {
    "Complaint ID": "complaint_id",
    "Consumer complaint narrative": "text",
    "Product": "topic",
    "Date received": "date_received",
    "Company": "company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_splits(
    df: pd.DataFrame, size: int, min_class: int = 100, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Sample by topic and split each topic into roughly 70/15/15 percent."""
    if size < 1 or min_class < 1:
        raise ValueError("Sample size and minimum class count must be positive")
    if len(df) < size:
        raise ValueError(f"Requested {size} rows, but only {len(df)} remain after cleaning")
    df = df.copy()
    counts = df["topic"].value_counts()
    df.loc[df["topic"].map(counts) < min_class, "topic"] = "Other"
    counts = df.groupby("topic", sort=True).size()
    quotas = counts * size / len(df)
    targets = np.floor(quotas).astype(int)
    extra = size - int(targets.sum())
    order = (quotas - targets).sort_values(ascending=False, kind="stable").index
    targets.loc[order[:extra]] += 1
    if (targets < 10).any():
        raise ValueError("Each topic needs at least 10 sampled rows; raise size or min_class")

    parts = {"train": [], "val": [], "test": []}
    for topic, group in df.groupby("topic", sort=True):
        sample = group.sample(n=int(targets[topic]), random_state=seed)
        n = len(sample)
        train_end = round(n * 0.70)
        val_end = train_end + round(n * 0.15)
        parts["train"].append(sample.iloc[:train_end])
        parts["val"].append(sample.iloc[train_end:val_end])
        parts["test"].append(sample.iloc[val_end:])
    return {
        name: pd.concat(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
        for name, rows in parts.items()
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_store(
    paths: list[Path], db_path: Path, year: int | None = None,
    chunksize: int = 10000, seed: int = SEED,
) -> dict:
    """Clean CSV chunks into a disk-backed table with sequential drop counts."""
    if db_path.exists():
        raise ValueError(f"Database already exists: {db_path}")
    reasons = ["missing_id", "missing_text", "missing_topic", "invalid_date",
               "outside_year", "duplicate_id", "duplicate_text"]
    stats = {"input_rows": 0, "kept_rows": 0, "dropped": dict.fromkeys(reasons, 0), "sources": []}
    cols = list(COLUMNS.values())
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA cache_size = -16000")
        conn.execute("CREATE TABLE seen_ids (id TEXT PRIMARY KEY)")
        fields = ", ".join(f"{col} TEXT" for col in cols)
        conn.execute(f"CREATE TABLE complaints ({fields}, text_hash TEXT UNIQUE, sample_key TEXT)")
        insert = "INSERT OR IGNORE INTO complaints VALUES (" + ",".join("?" for _ in range(len(cols) + 2)) + ")"
        for path in paths:
            source_rows = 0
            reader = pd.read_csv(path, usecols=lambda col: col in COLUMNS, dtype="string", chunksize=chunksize)
            for df in reader:
                df = df.rename(columns=COLUMNS)
                required = {"complaint_id", "text", "topic"} | ({"date_received"} if year else set())
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
                source_rows += len(df)
                stats["input_rows"] += len(df)
                for col, reason in [("complaint_id", "missing_id"), ("text", "missing_text"), ("topic", "missing_topic")]:
                    df[col] = df[col].fillna("").str.strip()
                    bad = df[col].eq("")
                    stats["dropped"][reason] += int(bad.sum())
                    df = df.loc[~bad].copy()
                if year:
                    dates = pd.to_datetime(df["date_received"], errors="coerce", format="mixed")
                    bad = dates.isna()
                    stats["dropped"]["invalid_date"] += int(bad.sum())
                    outside = dates.notna() & dates.dt.year.ne(year)
                    stats["dropped"]["outside_year"] += int(outside.sum())
                    df = df.loc[~(bad | outside)].copy()
                    df["date_received"] = dates.loc[df.index].dt.strftime("%Y-%m-%d")
                df["text"] = df["text"].map(clean_text)
                df = df.reindex(columns=cols).fillna("")
                for row in df.itertuples(index=False, name=None):
                    complaint_id, text = row[:2]
                    if not conn.execute("INSERT OR IGNORE INTO seen_ids VALUES (?)", (complaint_id,)).rowcount:
                        stats["dropped"]["duplicate_id"] += 1
                        continue
                    text_hash = hashlib.sha256(text.casefold().encode()).hexdigest()
                    key = hashlib.sha256(f"{seed}:{complaint_id}".encode()).hexdigest()
                    if conn.execute(insert, (*row, text_hash, key)).rowcount:
                        stats["kept_rows"] += 1
                    else:
                        stats["dropped"]["duplicate_text"] += 1
                conn.commit()
            stats["sources"].append({"file": path.name, "sha256": file_hash(path), "rows": source_rows})
            log.info("Read %s: %s rows; %s kept so far", path.name, source_rows, stats["kept_rows"])
        conn.execute("DROP TABLE seen_ids")
        conn.execute("CREATE INDEX complaint_topic ON complaints(topic)")
    assert stats["input_rows"] == stats["kept_rows"] + sum(stats["dropped"].values())
    return stats


def sample_store(db_path: Path, size: int, min_class: int) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        counts = pd.read_sql_query("SELECT topic, COUNT(*) AS n FROM complaints GROUP BY topic ORDER BY topic", conn)
        if size < 1 or min_class < 1:
            raise ValueError("Sample size and minimum class count must be positive")
        if counts["n"].sum() < size:
            raise ValueError("Not enough cleaned rows for the requested sample")
        counts["label"] = counts["topic"].where(counts["n"] >= min_class, "Other")
        totals = counts.groupby("label")["n"].sum()
        quotas = totals * size / totals.sum()
        targets = np.floor(quotas).astype(int)
        order = (quotas - targets).sort_values(ascending=False, kind="stable").index
        targets.loc[order[:size - int(targets.sum())]] += 1
        if (targets < 10).any():
            raise ValueError("Each topic needs at least 10 sampled rows; raise size or min_class")
        frames = []
        for label, n in targets.items():
            topics = counts.loc[counts["label"].eq(label), "topic"].tolist()
            placeholders = ",".join("?" for _ in topics)
            query = f"SELECT {', '.join(COLUMNS.values())} FROM complaints WHERE topic IN ({placeholders}) ORDER BY sample_key, complaint_id LIMIT ?"
            df = pd.read_sql_query(query, conn, params=topics + [int(n)])
            df["topic"] = label
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Prepare versioned CSV splits from a local CFPB export."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV, single-CSV ZIP, or directory of ZIP archives")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=30000)
    parser.add_argument("--min-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--year", type=int)
    parser.add_argument("--chunksize", type=int, default=10000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not 30000 <= args.size <= 50000:
        parser.error("--size must be between 30000 and 50000")
    if args.min_class < 1 or args.chunksize < 1:
        parser.error("--min-class and --chunksize must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to preserve splits")
    paths = sorted(args.input.glob("*.zip")) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error("No ZIP archives found")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    db_path = args.output / "cleaned.sqlite"
    stats = build_store(paths, db_path, args.year, args.chunksize, args.seed)
    sample = sample_store(db_path, args.size, args.min_class)
    splits = make_splits(sample, args.size, min_class=1, seed=args.seed)
    manifest = {
        "sources": stats["sources"],
        "year": args.year,
        "seed": args.seed,
        "sample_size": args.size,
        "min_class": args.min_class,
        "clean_rows": stats["kept_rows"],
        "drop_counts": stats["dropped"],
        "input_rows": stats["input_rows"],
        "dedup_order": "sorted archive filename, then CSV row order; first valid ID, then first normalized text",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "splits": {},
    }
    for name, split in splits.items():
        path = args.output / f"{name}.csv"
        split.to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": len(split),
            "topics": split["topic"].value_counts().to_dict(),
            "sha256": file_hash(path),
        }
        log.info("Saved %s rows to %s", len(split), path)
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Drop counts: %s", stats["dropped"])


if __name__ == "__main__":
    main()

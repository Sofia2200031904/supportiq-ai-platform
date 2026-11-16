import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.data_prep import COLUMNS, clean_text, file_hash


def flag_text(texts: pd.Series, seen: set[bytes]) -> tuple[list[bool], list[bool]]:
    present, repeated = [], []
    for text in texts:
        text = clean_text(text)
        present.append(bool(text))
        key = hashlib.sha256(text.casefold().encode()).digest() if text else None
        repeated.append(key is not None and key in seen)
        if key is not None:
            seen.add(key)
    return present, repeated


def prepare(input_dir: Path, output: Path, year: int = 2025) -> dict:
    """Keep source rows for analytics and flag repeated narrative text."""
    paths = sorted(input_dir.glob("*.zip"))
    if not paths:
        raise ValueError("No source ZIP files found")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analytics output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    stats = {"year": year, "rows": 0, "with_narrative": 0, "duplicate_text": 0, "months": {}, "sources": []}
    for path in paths:
        dest = output / f"{path.stem}.parquet"
        writer = None
        source_rows = 0
        try:
            for df in pd.read_csv(path, dtype="string", usecols=lambda col: col in COLUMNS, chunksize=10000):
                df = df.rename(columns=COLUMNS)
                required = {"complaint_id", "text", "topic", "date_received"}
                if not required.issubset(df.columns):
                    raise ValueError(f"Missing required columns in {path.name}")
                df = df.reindex(columns=COLUMNS.values()).fillna("")
                dates = pd.to_datetime(df["date_received"], errors="coerce", format="mixed")
                if dates.isna().any() or dates.dt.year.ne(year).any():
                    raise ValueError(f"Unexpected date in {path.name}; analytics keeps rows rather than silently dropping them")
                df["date_received"] = dates.dt.strftime("%Y-%m-%d")
                df["has_narrative"], df["is_duplicate_text"] = flag_text(df["text"], seen)
                source_rows += len(df)
                stats["rows"] += len(df)
                stats["with_narrative"] += int(df["has_narrative"].sum())
                stats["duplicate_text"] += int(df["is_duplicate_text"].sum())
                for month, n in dates.dt.strftime("%Y-%m").value_counts().items():
                    stats["months"][month] = stats["months"].get(month, 0) + int(n)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        stats["sources"].append({"file": path.name, "sha256": file_hash(path), "rows": source_rows,
                                 "output": dest.name, "output_sha256": file_hash(dest)})
        logging.info("Kept all %s rows from %s", source_rows, path.name)
    stats["months"] = dict(sorted(stats["months"].items()))
    (output / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/2025"))
    parser.add_argument("--output", type=Path, default=Path("data/analytics-2025"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stats = prepare(args.input, args.output)
    logging.info("Saved %s rows, including %s repeated narratives", stats["rows"], stats["duplicate_text"])


if __name__ == "__main__":
    main()

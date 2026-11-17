import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from data.data_prep import SEED, file_hash, make_splits

log = logging.getLogger(__name__)
NAMES = ("train", "val", "test")


def find_pairs(x, threshold=0.9, batch_size=256):
    """Return every upper-triangle pair above the cosine threshold."""
    right = x.T.tocsr()
    rows, cols, scores = [], [], []
    for start in range(0, x.shape[0], batch_size):
        # No top-k truncation: every column is eligible for retention.
        matches = sp_matmul_topn(x[start:start + batch_size], right,
                                top_n=x.shape[0], threshold=threshold,
                                density=0.001, n_threads=2).tocoo()
        left = matches.row + start
        keep = (left < matches.col) & (matches.data > threshold)
        rows.extend(left[keep])
        cols.extend(matches.col[keep])
        scores.extend(matches.data[keep])
        if start % (batch_size * 20) == 0:
            log.info("Compared %s / %s rows", min(start + batch_size, x.shape[0]), x.shape[0])
    return pd.DataFrame({"left": rows, "right": cols, "similarity": scores})


def remove_neighbors(df, pairs, seed=SEED):
    neighbors = [[] for _ in range(len(df))]
    for row in pairs.itertuples(index=False):
        neighbors[row.left].append((row.right, row.similarity))
        neighbors[row.right].append((row.left, row.similarity))
    order = sorted(range(len(df)), key=lambda i: hashlib.sha256(f"{seed}:{df.iloc[i]['complaint_id']}".encode()).digest())
    removed = {}
    for i in order:
        if i in removed:
            continue
        for j, score in neighbors[i]:
            if j not in removed:
                removed[j] = (i, score)
    kept = sorted(set(range(len(df))) - set(removed))
    return kept, removed


def merge_small_classes(df, min_test=50):
    df = df.copy()
    mapping = {label: label for label in sorted(df["topic"].unique())}
    while True:
        counts = df["topic"].value_counts()
        test_counts = counts - (counts * .70).round().astype(int) - (counts * .15).round().astype(int)
        small = test_counts[test_counts < min_test].index.tolist()
        if not small:
            return df, mapping
        if len(counts) == 1:
            raise ValueError("Too few rows to provide the minimum test count")
        if small == ["Other"]:
            small.append(counts.drop("Other").sort_values(kind="stable").index[0])
        for label in small:
            df.loc[df["topic"].eq(label), "topic"] = "Other"
            for original, target in mapping.items():
                if target == label:
                    mapping[original] = "Other"


def counts_table(splits):
    return pd.DataFrame({name: df["topic"].value_counts() for name, df in splits.items()}).fillna(0).astype(int).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed-2025"))
    parser.add_argument("--output", type=Path, default=Path("data/processed-2025-v2"))
    parser.add_argument("--reports", type=Path, default=Path("reports/split_audit"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    source_manifest = json.loads((args.input / "manifest.json").read_text())
    original = {}
    for name in NAMES:
        path = args.input / f"{name}.csv"
        if file_hash(path) != source_manifest["splits"][name]["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
        original[name] = pd.read_csv(path, dtype="string", keep_default_na=False)
    df = pd.concat([original[name].assign(old_split=name) for name in NAMES], ignore_index=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, norm="l2", dtype=np.float64)
    log.info("Vectorizing %s narratives", len(df))
    x = vectorizer.fit_transform(df["text"])
    if np.any(np.diff(x.indptr) == 0):
        raise ValueError("Some narratives have no word features; review them before auditing")
    pairs = find_pairs(x)
    pairs["left_split"] = df.loc[pairs["left"], "old_split"].to_numpy()
    pairs["right_split"] = df.loc[pairs["right"], "old_split"].to_numpy()
    cross = pairs[pairs["left_split"] != pairs["right_split"]]
    kept, removed = remove_neighbors(df, pairs[["left", "right", "similarity"]])
    kept_set = set(kept)
    assert not any(a in kept_set and b in kept_set for a, b in pairs[["left", "right"]].itertuples(index=False, name=None))
    cleaned, mapping = merge_small_classes(df.iloc[kept].drop(columns="old_split"))
    splits = make_splits(cleaned, len(cleaned), min_class=1)
    after = counts_table(splits)
    assert after["test"].min() >= 50
    log.info("Verifying retained narratives with a second exhaustive pass")
    remaining = find_pairs(x[kept])
    assert remaining.empty, "Near-duplicate survived removal"
    args.output.mkdir(parents=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    before = counts_table(original)
    before.to_csv(args.reports / "class_counts_before.csv", index_label="topic")
    after.to_csv(args.reports / "class_counts_after.csv", index_label="topic")
    removed_rows = pd.DataFrame([
        {"complaint_id": df.iloc[i]["complaint_id"], "old_split": df.iloc[i]["old_split"],
         "kept_id": df.iloc[j]["complaint_id"], "similarity": score}
        for i, (j, score) in sorted(removed.items())
    ], columns=["complaint_id", "old_split", "kept_id", "similarity"])
    removed_rows.to_csv(args.reports / "removed_rows.csv", index=False)
    pairs["left_id"] = df.loc[pairs["left"], "complaint_id"].to_numpy()
    pairs["right_id"] = df.loc[pairs["right"], "complaint_id"].to_numpy()
    pairs.drop(columns=["left", "right"]).to_csv(args.reports / "near_duplicate_pairs.csv.gz", index=False)
    sparse.save_npz(args.output / "audit_vectors.npz", x[kept])
    df.iloc[kept][["complaint_id"]].to_csv(args.output / "audit_row_ids.csv", index=False)
    audit = {
        "similarity": "L2-normalized word unigram/bigram TF-IDF cosine; default sklearn word tokenization; no stopword, document-frequency or vocabulary pruning",
        "threshold": 0.9, "comparison": "strictly greater than", "scope": "all pairs in original 30,000-row sample; not the full 607,931-row pool",
        "audit_idf": "Fitted once on original sample, frozen for verification; never reused as model preprocessing",
        "source_manifest_sha256": file_hash(args.input / "manifest.json"),
        "original_rows": len(df), "all_near_pairs": len(pairs), "cross_split_pairs_before": len(cross),
        "cross_split_rows_before": len(set(cross["left"]) | set(cross["right"])),
        "pairs_by_original_split": {f"{a}/{b}": int(n) for (a, b), n in cross.groupby(["left_split", "right_split"]).size().items()},
        "removed_rows": len(removed), "removed_by_old_split": removed_rows["old_split"].value_counts().to_dict(),
        "retained_rows": len(cleaned), "near_pairs_after": len(remaining), "cross_split_pairs_after": 0,
        "removal_rule": "Seeded ID order: keep each available row, discard its direct near neighbors; includes within-split pairs to keep rebuilding safe",
        "label_mapping": mapping, "minimum_test_examples": 50,
        "class_counts_before": before.to_dict(orient="index"), "class_counts_after": after.to_dict(orient="index"),
    }
    manifest = {"seed": SEED, "sample_size": len(cleaned), "source_directory": str(args.input),
                "audit": audit, "splits": {}}
    for name, frame in splits.items():
        path = args.output / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["splits"][name] = {"rows": len(frame), "topics": frame["topic"].value_counts().to_dict(), "sha256": file_hash(path)}
    manifest["audit_vectors_sha256"] = file_hash(args.output / "audit_vectors.npz")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.reports / "audit.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Removed %s rows; retained %s; cross-split pairs after: 0", len(removed), len(cleaned))


if __name__ == "__main__":
    main()

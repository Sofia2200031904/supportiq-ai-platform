import numpy as np
import pandas as pd
import json
from pathlib import Path
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from data.refine_splits import find_pairs, merge_small_classes, remove_neighbors
from data.data_prep import file_hash


def test_pair_search_matches_dense_cosine_without_top_k_limit():
    texts = ["bank charged a fee on my account " * 4 + str(i) for i in range(12)] + ["mortgage payment delayed", "credit report incorrect"]
    x = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
    pairs = find_pairs(x, batch_size=3)
    scores = (x @ x.T).toarray()
    expected = {(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts)) if scores[i, j] > .9}
    assert set(pairs[["left", "right"]].itertuples(index=False, name=None)) == expected
    assert len(expected) > 10


def test_threshold_is_strict():
    x = sparse.csr_matrix([[1., 0.], [.9, np.sqrt(1 - .9**2)]])
    assert find_pairs(x).empty


def test_removal_keeps_no_linked_rows_and_is_repeatable():
    df = pd.DataFrame({"complaint_id": list("abcde")})
    pairs = pd.DataFrame({"left": [0, 1, 3], "right": [1, 2, 4], "similarity": [.95, .92, .99]})
    kept, removed = remove_neighbors(df, pairs)
    assert (kept, removed) == remove_neighbors(df, pairs)
    assert len(kept) + len(removed) == len(df)
    assert all(not (a in kept and b in kept) for a, b in zip(pairs.left, pairs.right))
    assert all(parent in kept and score > .9 for parent, score in removed.values())


def test_other_also_meets_minimum_test_size():
    df = pd.DataFrame({"topic": ["A"] * 1000 + ["B"] * 400 + ["C"] * 190 + ["D"] * 126})
    result, mapping = merge_small_classes(df)
    assert mapping == {"A": "A", "B": "Other", "C": "Other", "D": "Other"}
    assert result.topic.value_counts().to_dict() == {"A": 1000, "Other": 716}


def test_saved_refined_splits():
    folder = Path(__file__).resolve().parents[1] / "data" / "processed-2025-v2"
    if not (folder / "manifest.json").exists():
        pytest.skip("Run split refinement first")
    manifest = json.loads((folder / "manifest.json").read_text())
    assert file_hash(folder / "audit_vectors.npz") == manifest["audit_vectors_sha256"]
    ids = pd.read_csv(folder / "audit_row_ids.csv", dtype=str)["complaint_id"]
    all_ids = []
    for name in ("train", "val", "test"):
        df = pd.read_csv(folder / f"{name}.csv", usecols=["complaint_id", "topic"], dtype=str)
        all_ids.extend(df["complaint_id"])
        if name == "test":
            assert df["topic"].value_counts().min() >= 50
    assert set(ids) == set(all_ids)
    assert len(ids) == len(all_ids) == manifest["sample_size"]
    # Recheck the saved frozen audit vectors rather than trusting reported zeros.
    x = sparse.load_npz(folder / "audit_vectors.npz")
    assert x.shape[0] == len(ids)
    assert find_pairs(x).empty

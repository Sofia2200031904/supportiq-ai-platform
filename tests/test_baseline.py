import json

import joblib
import numpy as np
import pandas as pd
import pytest

from data.data_prep import file_hash
from ml.baseline import build_pipeline, choose_best
from ml.data import load_splits, verify_splits


@pytest.mark.parametrize("kind", ["logreg", "svm"])
def test_pipeline_training_only_vocabulary_and_reload(kind, tmp_path):
    texts = ["bank account fee", "checking account charge", "mortgage home loan", "mortgage interest loan"] * 3
    labels = ["Account", "Account", "Mortgage", "Mortgage"] * 3
    model = build_pipeline(kind, 1, min_df=1, max_features=100)
    model.fit(texts, labels)
    assert model.named_steps["classifier"].class_weight == "balanced"
    assert model.named_steps["tfidf"].ngram_range == (1, 2)
    pred = model.predict(["unseenmarker account fee", "mortgage loan"])
    assert "unseenmarker" not in model.named_steps["tfidf"].vocabulary_
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(pred, joblib.load(path).predict(["unseenmarker account fee", "mortgage loan"]))


def test_model_selection_ignores_test_results():
    trials = [{"C": 2, "val_macro_f1": .6, "test_macro_f1": .9},
              {"C": 1, "val_macro_f1": .7, "test_macro_f1": .5},
              {"C": .5, "val_macro_f1": .7, "test_macro_f1": .1}]
    assert choose_best(trials)["C"] == .5


def test_split_hashes_are_anchored_outside_data_directory(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    spec = {"splits": {}}
    for name in ("train", "val", "test"):
        path = folder / f"{name}.csv"
        pd.DataFrame({"complaint_id": [name], "text": ["bank fee"], "topic": ["Account"]}).to_csv(path, index=False)
        spec["splits"][name] = {"sha256": file_hash(path), "rows": 1, "topics": {"Account": 1}}
    anchor = tmp_path / "frozen.json"
    anchor.write_text(json.dumps(spec))
    frames, _ = load_splits(folder, anchor)
    assert len(frames["train"]) == 1
    (folder / "test.csv").write_text("changed")
    (folder / "manifest.json").write_text(json.dumps({"splits": {"test": {"sha256": file_hash(folder / "test.csv")}}}))
    with pytest.raises(ValueError, match="Frozen split hash mismatch: test"):
        verify_splits(folder, anchor)

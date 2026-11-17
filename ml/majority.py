import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from data.data_prep import file_hash
from ml.evaluate import PRIMARY_METRIC, class_weights, score_predictions

DATA_DIR = Path("data/processed-2025-v2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("reports/split_audit/majority_baseline.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    labels = {}
    for name in ("train", "val", "test"):
        path = args.data / f"{name}.csv"
        if file_hash(path) != manifest["splits"][name]["sha256"]:
            raise ValueError(f"Split hash mismatch: {name}")
        labels[name] = pd.read_csv(path, usecols=["topic"])["topic"].to_numpy()
    train = labels["train"]
    classes = np.unique(train).tolist()
    model = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), train)
    report = {"primary_metric": PRIMARY_METRIC, "strategy": "most_frequent", "data_directory": str(args.data),
              "manifest_sha256": file_hash(args.data / "manifest.json"),
              "majority_class": str(model.predict([[0]])[0]), "class_weights_from_train": class_weights(train),
              "weight_policy": "Balanced weights for trainable models; majority control intentionally uses original training frequencies",
              "scores": {}}
    for name in ("val", "test"):
        pred = model.predict(np.zeros((len(labels[name]), 1)))
        report["scores"][name] = score_predictions(labels[name], pred, classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in scores.items() if k != "per_class"} for name, scores in report["scores"].items()}, indent=2))


if __name__ == "__main__":
    main()

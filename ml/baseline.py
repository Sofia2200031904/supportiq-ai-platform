import argparse
import json
import logging
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from data.data_prep import SEED, file_hash
from ml.data import DATA_DIR, ROOT, load_splits, verify_splits
from ml.evaluate import PRIMARY_METRIC, score_predictions

log = logging.getLogger(__name__)


def build_pipeline(kind, c, max_features=50000, min_df=2):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features,
                          sublinear_tf=True, dtype=np.float64)
    if kind == "logreg":
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEED)
    elif kind == "svm":
        model = LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=5000, random_state=SEED)
    else:
        raise ValueError(f"Unknown classifier: {kind}")
    return Pipeline([("tfidf", tfidf), ("classifier", model)])


def choose_best(trials):
    # Stable tie-break: prefer the smaller C; test scores are never consulted.
    return max(trials, key=lambda trial: (trial["val_macro_f1"], -trial["C"]))


def benchmark(model, texts, verify):
    texts = texts.iloc[:1000]
    verify()
    model.predict(texts.iloc[:10])
    times = []
    for _ in range(3):
        verify()
        start = time.perf_counter()
        model.predict(texts)
        times.append((time.perf_counter() - start) * 1000 / len(texts))
    return {"seconds_per_1000": float(np.median(times)), "timing_repeats": times,
            "timing_rows": len(texts), "timing_method": "median of 3 warm calls; includes TF-IDF and prediction; excludes loading and hash checks; 2 CPU threads"}


def train(args):
    """Select classic models on validation data, then evaluate each winner once."""
    frames, manifest = load_splits(args.data)
    train_df, val, test = (frames[name] for name in ("train", "val", "test"))
    labels = sorted(train_df["topic"].unique())
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Results directory is not empty; choose a new --output")
    if args.models.exists() and any(args.models.iterdir()):
        raise ValueError("Model directory is not empty; choose a new --models")
    args.output.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    tracking = ROOT / "artifacts" / "mlflow"
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri or f"sqlite:///{(tracking / 'tracking.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name("supportiq-classic")
    if experiment is None:
        mlflow.create_experiment("supportiq-classic", artifact_location=(tracking / "runs").as_uri())
    mlflow.set_experiment("supportiq-classic")
    results = {"primary_metric": PRIMARY_METRIC, "selection": "validation macro-F1 only; no refit on validation",
               "split_hashes": {name: spec["sha256"] for name, spec in manifest["splits"].items()},
               "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "mlflow": mlflow.__version__},
               "timing_hardware": {"platform": platform.platform(), "processor": platform.processor(), "threads": 2},
               "trials": [], "models": {}}
    selected = {}
    with threadpool_limits(limits=2):
        for kind in ("logreg", "svm"):
            trials = []
            best_model, best_score = None, -1
            for c in (.5, 1., 2.):
                verify_splits(args.data)
                model = build_pipeline(kind, c)
                log.info("Training %s with C=%s", kind, c)
                with mlflow.start_run(run_name=f"{kind}-C{c}") as run:
                    mlflow.log_params({"classifier": kind, "C": c, "ngram_range": "1,2", "max_features": 50000,
                                       "min_df": 2, "sublinear_tf": True, "class_weight": "balanced", "seed": SEED,
                                       "primary_metric": PRIMARY_METRIC, "train_rows": len(train_df), "val_rows": len(val)})
                    mlflow.set_tags({f"{name}_sha256": spec["sha256"] for name, spec in manifest["splits"].items()})
                    start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(train_df["text"], train_df["topic"])
                    fit_seconds = time.perf_counter() - start
                    verify_splits(args.data)
                    pred = model.predict(val["text"])
                    metrics = score_predictions(val["topic"], pred, labels)
                    trial = {"model": kind, "C": c, "val_macro_f1": metrics["macro_f1"],
                             "val_accuracy": metrics["accuracy"], "fit_seconds": fit_seconds, "run_id": run.info.run_id}
                    mlflow.log_metrics({key: trial[key] for key in ("val_macro_f1", "val_accuracy", "fit_seconds")})
                    trials.append(trial)
                    results["trials"].append(trial)
                    if metrics["macro_f1"] > best_score:
                        best_model, best_score = model, metrics["macro_f1"]
                    log.info("%s C=%s validation macro-F1 %.4f", kind, c, metrics["macro_f1"])
            best = choose_best(trials)
            selected[kind] = (best_model, best)
        results["recommended_model"] = max(selected, key=lambda name: selected[name][1]["val_macro_f1"])
        # Hyperparameters and the overall recommendation are now locked before test access.
        for kind, (model, best) in selected.items():
            verify_splits(args.data)
            pred = model.predict(test["text"])
            metrics = score_predictions(test["topic"], pred, labels)
            metrics.update(benchmark(model, test["text"], lambda: verify_splits(args.data)))
            cm = confusion_matrix(test["topic"], pred, labels=labels)
            metrics.update({"C": best["C"], "val_macro_f1": best["val_macro_f1"], "run_id": best["run_id"],
                            "labels": labels, "confusion_matrix": cm.tolist(), "rows": len(test),
                            "cost_per_1000_usd": None, "cost_note": "Local CPU execution; hosting cost not measured"})
            path = args.models / f"{kind}.joblib"
            joblib.dump(model, path, compress=3)
            reloaded = joblib.load(path)
            verify_splits(args.data)
            if not np.array_equal(pred, reloaded.predict(test["text"])):
                raise RuntimeError("Reloaded model predictions changed")
            metrics["model_sha256"] = file_hash(path)
            results["models"][kind] = metrics
            pd.DataFrame({"complaint_id": test["complaint_id"], "actual": test["topic"], "predicted": pred}).to_csv(args.output / f"{kind}_predictions.csv", index=False)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(args.output / f"{kind}_confusion.csv")
            per_class = pd.DataFrame({label: metrics["per_class"][label] for label in labels}).T
            per_class.to_csv(args.output / f"{kind}_per_class.csv", index_label="topic")
            fig, ax = plt.subplots(figsize=(12, 10))
            short = ["Checking/savings", "Credit card", "Credit reporting", "Debt collection", "Money transfers", "Mortgage", "Other", "Student loan", "Vehicle loan"]
            ConfusionMatrixDisplay(cm, display_labels=short).plot(ax=ax, xticks_rotation=55, colorbar=False, cmap="Blues")
            ax.set_title(f"{kind}: frozen test set")
            fig.tight_layout()
            fig.savefig(args.output / f"{kind}_confusion.png", dpi=120)
            plt.close(fig)
            with mlflow.start_run(run_id=best["run_id"]):
                mlflow.log_metrics({"test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"], "seconds_per_1000": metrics["seconds_per_1000"]})
                mlflow.log_dict(metrics, "test_metrics.json")
                mlflow.log_artifact(str(path), "model")
            log.info("%s test macro-F1 %.4f; accuracy %.4f", kind, metrics["macro_f1"], metrics["accuracy"])
    joblib.dump(selected[results["recommended_model"]][0], args.models / "best.joblib", compress=3)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.models / "metadata.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(results["trials"]).to_csv(args.output / "validation_runs.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "baseline")
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts" / "models" / "baseline")
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(args)


if __name__ == "__main__":
    main()

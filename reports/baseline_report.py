import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "reports" / "baseline"
    result = json.loads((folder / "results.json").read_text())
    majority = json.loads((root / "reports" / "split_audit" / "majority_baseline.json").read_text())
    names = ["Majority", "Logistic Regression", "Linear SVM"]
    models = [majority["scores"]["test"], result["models"]["logreg"], result["models"]["svm"]]
    lines = ["# Classic ML baseline", "", "These results use the unchanged 17,798 / 3,814 / 3,813 splits and all nine classes.", "",
             "| Model | Validation macro-F1 | Test macro-F1 | Test accuracy | Seconds / 1,000 texts |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, model in zip(names, models):
        val = f"{model['val_macro_f1']:.4f}" if "val_macro_f1" in model else f"{majority['scores']['val']['macro_f1']:.4f}"
        speed = f"{model['seconds_per_1000']:.3f}" if "seconds_per_1000" in model else "Not measured"
        lines.append(f"| {name} | {val} | {model['macro_f1']:.4f} | {model['accuracy']:.2%} | {speed} |")
    lines.extend(["", f"Validation selects **{result['recommended_model']}**. This choice was made before reading test scores.", "",
                  "## Search and artifacts", "", "Each algorithm tried C = 0.5, 1, and 2. Both use training-only word/bigram TF-IDF and balanced class weights. The vocabulary is capped at 50,000 features with min_df=2 and sublinear term frequency. No refit on validation was performed.", ""])
    for kind, model in result["models"].items():
        lines.append(f"- {kind}: selected C={model['C']}; MLflow run `{model['run_id']}`. Per-class scores: `{kind}_per_class.csv`. Confusion matrix: `{kind}_confusion.png`.")
    lines.extend(["", "All six trials are logged in local MLflow and `validation_runs.csv`. Saved pipelines and the validation-selected `best.joblib` are under `artifacts/models/baseline`. Reloaded pipelines produced the same test predictions.", "",
                  "## Timing and limits", "", "Timing includes TF-IDF preprocessing and classifier inference on 1,000 fixed test texts. It is the median of three warm runs, with two CPU threads. Loading and split-hash checks are outside the timer. See `results.json` for timings, hardware, versions, and split hashes. Cloud cost was not measured.", "",
                  "The test set was not used to choose hyperparameters. These are random splits within 2025, not a future-period holdout. Lexical classifiers can miss meaning expressed with different wording; SVM decision scores are not calibrated probabilities. The merged Other label combines several products.", ""])
    (folder / "results.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    values = [model["macro_f1"] for model in models]
    bars = ax.barh(names[::-1], values[::-1], color=["#d9982d", "#16775f", "#8b9ba4"])
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values[::-1]], padding=5)
    ax.set(xlim=(0, 1), xlabel="Macro-F1", title="Classic models on the frozen test set")
    fig.tight_layout()
    fig.savefig(folder / "macro_f1.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

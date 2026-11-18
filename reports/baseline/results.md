# Classic ML baseline

These results use the unchanged 17,798 / 3,814 / 3,813 splits and all nine classes.

| Model | Validation macro-F1 | Test macro-F1 | Test accuracy | Seconds / 1,000 texts |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.0773 | 0.0774 | 53.42% | Not measured |
| Logistic Regression | 0.7290 | 0.7299 | 80.99% | 1.029 |
| Linear SVM | 0.7258 | 0.7301 | 81.77% | 0.979 |

Validation selects **logreg**. This choice was made before reading test scores.

## Search and artifacts

Each algorithm tried C = 0.5, 1, and 2. Both use training-only word/bigram TF-IDF and balanced class weights. The vocabulary is capped at 50,000 features with min_df=2 and sublinear term frequency. No refit on validation was performed.

- logreg: selected C=2.0; MLflow run `cca6d9d7094742c59940bb123e00260b`. Per-class scores: `logreg_per_class.csv`. Confusion matrix: `logreg_confusion.png`.
- svm: selected C=0.5; MLflow run `5bfc153e70d0463fb224ed251f2b1a3a`. Per-class scores: `svm_per_class.csv`. Confusion matrix: `svm_confusion.png`.

All six trials are logged in local MLflow and `validation_runs.csv`. Saved pipelines and the validation-selected `best.joblib` are under `artifacts/models/baseline`. Reloaded pipelines produced the same test predictions.

## Timing and limits

Timing includes TF-IDF preprocessing and classifier inference on 1,000 fixed test texts. It is the median of three warm runs, with two CPU threads. Loading and split-hash checks are outside the timer. See `results.json` for timings, hardware, versions, and split hashes. Cloud cost was not measured.

The test set was not used to choose hyperparameters. These are random splits within 2025, not a future-period holdout. Lexical classifiers can miss meaning expressed with different wording; SVM decision scores are not calibrated probabilities. The merged Other label combines several products.

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

PRIMARY_METRIC = "macro_f1"


def score_predictions(y_true, y_pred, labels):
    return {
        PRIMARY_METRIC: float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def class_weights(y_train):
    labels = np.unique(y_train)
    values = compute_class_weight("balanced", classes=labels, y=np.asarray(y_train))
    return dict(zip(labels.tolist(), values.tolist()))


def fit_weighted(model, x_train, y_train):
    """Fit a supervised sklearn classifier with training-only class balancing."""
    if model.get_params().get("class_weight") is not None:
        raise ValueError("Set class_weight=None when using this helper to avoid double weighting")
    return model.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml.evaluate import class_weights, fit_weighted, score_predictions


def test_macro_f1_penalizes_majority_only_prediction():
    scores = score_predictions(["A"] * 9 + ["B"], ["A"] * 10, ["A", "B"])
    assert scores["accuracy"] == .9
    assert scores["macro_f1"] == pytest.approx((18 / 19) / 2)
    assert class_weights(["A"] * 9 + ["B"]) == {"A": 10 / 18, "B": 5.0}


def test_fit_applies_balanced_training_weights():
    x = np.array([[0], [1], [2], [3]])
    y = ["A", "A", "A", "B"]
    actual = fit_weighted(LogisticRegression(), x, y)
    expected = LogisticRegression(class_weight="balanced").fit(x, y)
    np.testing.assert_allclose(actual.coef_, expected.coef_)
    with pytest.raises(ValueError, match="double weighting"):
        fit_weighted(LogisticRegression(class_weight="balanced"), x, y)

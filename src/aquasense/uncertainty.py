"""
Uncertainty quantification module for AquaSense AI.
Provides Platt scaling probability calibration and Inductive Conformal Prediction (LAC).
"""

import numpy as np


class ConformalCalibrator:
    """
    Inductive Conformal Prediction (Least Ambiguous Classifier).
    Guarantees finite-sample distribution-free marginal coverage: P(Y in C(X)) >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.quantile_: float = 1.0
        self.calibrated_: bool = False

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray) -> "ConformalCalibrator":
        """
        Computes the non-conformity quantile on a calibration split.
        Non-conformity score: s_i = 1 - P(Y_i = y_i | x_i)
        """
        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Extract true class probabilities
        true_class_probs = probs[np.arange(n), y_true.astype(int)]
        non_conformity_scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile level
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_ = float(np.quantile(non_conformity_scores, q_level, method="higher"))
        self.calibrated_ = True
        return self

    def predict_set(self, prob_row: np.ndarray) -> list[int]:
        """
        Generates the prediction set for a single sample probability vector:
        C(x) = {y : P(y | x) >= 1 - q_hat}
        """
        threshold = max(0.0, 1.0 - self.quantile_)
        candidate_classes = [int(c) for c, p in enumerate(prob_row) if p >= threshold]
        # Return at least the argmax if empty due to boundary conditions
        if not candidate_classes:
            candidate_classes = [int(np.argmax(prob_row))]
        return sorted(candidate_classes)

    def evaluate_coverage(self, probs: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Evaluates empirical coverage and average set size on test data."""
        n = len(y_true)
        covered = 0
        set_sizes: list[int] = []

        for i in range(n):
            c_set = self.predict_set(probs[i])
            set_sizes.append(len(c_set))
            if int(y_true[i]) in c_set:
                covered += 1

        empirical_coverage = float(covered / n) if n > 0 else 0.0
        avg_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0

        return {
            "target_coverage": float(1.0 - self.alpha),
            "empirical_coverage": empirical_coverage,
            "average_set_size": avg_set_size,
            "quantile_threshold": float(self.quantile_),
        }


def assess_prediction_confidence(
    probs: np.ndarray,
    prediction_set: list[int],
    min_prob_threshold: float = 0.60,
) -> tuple[bool, str]:
    """
    Evaluates whether a prediction has high confidence or requires certified laboratory testing.
    Triggers 'Needs laboratory test' if:
    1. Prediction set contains multiple classes (|C(x)| > 1).
    2. Highest calibrated probability is below min_prob_threshold.
    """
    max_prob = float(np.max(probs))
    is_ambiguous = len(prediction_set) > 1
    is_low_prob = max_prob < min_prob_threshold

    if is_ambiguous or is_low_prob:
        reason = "Multiple plausible risk tiers detected." if is_ambiguous else "Low prediction confidence."
        return True, f"Screening support only. Needs laboratory test ({reason})"

    return False, "Sufficient screening confidence."

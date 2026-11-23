"""Inductive Conformal Prediction (ICP) for distribution-free uncertainty quantification."""

import numpy as np

from src.common.logger import logger


class ConformalPredictor:
    """Computes distribution-free conformal prediction sets with finite-sample coverage guarantees."""

    def __init__(self, calibration_scores: np.ndarray, classes: list[str] | None = None):
        """Initializes with sorted non-conformity scores from the calibration set."""
        self.calibration_scores = np.sort(np.asarray(calibration_scores))
        self.n_calib = len(self.calibration_scores)
        self.classes = classes or ["Good", "Moderate", "Polluted"]

    def get_quantile(self, alpha: float = 0.10) -> float:
        """Calculates conformal quantile threshold with finite-sample correction."""
        # Index: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((self.n_calib + 1) * (1.0 - alpha)) / self.n_calib)
        q_idx = int(np.clip(np.ceil(level * self.n_calib) - 1, 0, self.n_calib - 1))
        return float(self.calibration_scores[q_idx])

    def predict_set(self, probabilities: np.ndarray, alpha: float = 0.10) -> tuple[list[int], list[str]]:
        """Generates conformal prediction set for a single sample's probability distribution.
        
        Returns:
            (class_indices, class_names)
        """
        q_hat = self.get_quantile(alpha)
        threshold = 1.0 - q_hat

        # A class is included if its predicted probability is >= (1 - q_hat)
        included_indices = [int(c) for c, p in enumerate(probabilities) if p >= threshold]

        # If set is empty (extreme outlier or very high confidence threshold), include argmax class as fallback
        if not included_indices:
            included_indices = [int(np.argmax(probabilities))]

        included_names = [self.classes[idx] for idx in included_indices]
        return included_indices, included_names

    def predict_sets_batch(self, probabilities_matrix: np.ndarray, alpha: float = 0.10) -> list[list[str]]:
        """Vectorized generation of conformal prediction sets across multiple samples."""
        q_hat = self.get_quantile(alpha)
        threshold = 1.0 - q_hat

        batch_sets = []
        for probs in probabilities_matrix:
            indices = [int(c) for c, p in enumerate(probs) if p >= threshold]
            if not indices:
                indices = [int(np.argmax(probs))]
            batch_sets.append([self.classes[i] for i in indices])

        return batch_sets

    def evaluate_coverage(
        self,
        test_probabilities: np.ndarray,
        y_true: np.ndarray,
        alpha: float = 0.10
    ) -> dict[str, float]:
        """Evaluates empirical coverage and set efficiency on holdout test set."""
        q_hat = self.get_quantile(alpha)
        threshold = 1.0 - q_hat

        covered_count = 0
        set_sizes = []

        for i, probs in enumerate(test_probabilities):
            true_label = int(y_true[i])
            pred_indices = [int(c) for c, p in enumerate(probs) if p >= threshold]
            if not pred_indices:
                pred_indices = [int(np.argmax(probs))]

            set_sizes.append(len(pred_indices))
            if true_label in pred_indices:
                covered_count += 1

        n_test = len(y_true)
        empirical_coverage = covered_count / n_test if n_test > 0 else 0.0
        avg_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0

        logger.info(
            "Conformal evaluation complete",
            target_coverage=round(1.0 - alpha, 2),
            empirical_coverage=round(empirical_coverage, 4),
            avg_set_size=round(avg_set_size, 2)
        )

        return {
            "nominal_confidence": float(1.0 - alpha),
            "empirical_coverage": float(empirical_coverage),
            "mean_prediction_set_size": float(avg_set_size),
            "quantile_threshold": float(q_hat)
        }

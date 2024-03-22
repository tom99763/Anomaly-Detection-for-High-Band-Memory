import logging

import torch
from torchmetrics import Metric
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve as _BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_update,
)


class BinaryPrecisionRecallCurve(_BinaryPrecisionRecallCurve):
    """Binary precision-recall curve with without threshold prediction normalization."""
    @staticmethod
    def _binary_precision_recall_curve_format(
        preds: Tensor,
        target: Tensor,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Similar to torchmetrics' ``_binary_precision_recall_curve_format`` except it does not apply sigmoid."""
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]

        thresholds = _adjust_threshold_arg(thresholds, preds.device)
        return preds, target, thresholds

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with new predictions and targets.

        Unlike the base class, this accepts raw predictions and targets.

        Args:
            preds (Tensor): Predicted probabilities
            target (Tensor): Ground truth labels
        """
        preds, target, _ = BinaryPrecisionRecallCurve._binary_precision_recall_curve_format(
            preds,
            target,
            self.thresholds,
            self.ignore_index,
        )
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])


class F1AdaptiveThreshold(BinaryPrecisionRecallCurve):
    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)
        self.value = torch.tensor(default_value)

    def compute(self) -> torch.Tensor:

        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value


def normalize(anomaly_map, min_val, max_val, threshold = 0.5):
    normalized = ((anomaly_map - threshold) / (max_val - min_val)) + 0.5
    normalized = torch.minimum(normalized, torch.tensor(1.))
    normalized = torch.maximum(normalized, torch.tensor(0.))
    return normalized


class MinMax(Metric):
    """Track the min and max values of the observations in each batch.

    Args:
        full_state_update (bool, optional): Whether to update the state with the
            new values.
            Defaults to ``True``.
        kwargs: Any keyword arguments.

    Examples:
        >>> from anomalib.metrics import MinMax
        >>> import torch
        ...
        >>> predictions = torch.tensor([0.0807, 0.6329, 0.0559, 0.9860, 0.3595])
        >>> minmax = MinMax()
        >>> minmax(predictions)
        (tensor(0.0559), tensor(0.9860))

        It is possible to update the minmax values with a new tensor of predictions.

        >>> new_predictions = torch.tensor([0.3251, 0.3169, 0.3072, 0.6247, 0.9999])
        >>> minmax.update(new_predictions)
        >>> minmax.compute()
        (tensor(0.0559), tensor(0.9999))
    """

    full_state_update: bool = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)
        self.add_state("max", torch.tensor(float("-inf")), persistent=True)

        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))

    def update(self, predictions: torch.Tensor, *args, **kwargs) -> None:
        """Update the min and max values."""
        del args, kwargs  # These variables are not used.

        self.max = torch.max(self.max, torch.max(predictions))
        self.min = torch.min(self.min, torch.min(predictions))

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return min and max values."""
        return self.min, self.max
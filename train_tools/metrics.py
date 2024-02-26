import torch
from torchmetrics import ROC, PrecisionRecallCurve, Metric
from torchmetrics.functional import auc

class AUROC(ROC):
    def compute(self) -> torch.Tensor:
        tpr: torch.Tensor
        fpr: torch.Tensor
        fpr, tpr = self._compute()
        return auc(fpr, tpr, reorder=True)
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(preds.flatten(), target.flatten())
    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        tpr: torch.Tensor
        fpr: torch.Tensor
        fpr, tpr, _thresholds = super().compute()
        return (fpr, tpr)

class AUPR(PrecisionRecallCurve):
    def compute(self) -> torch.Tensor:
        """First compute PR curve, then compute area under the curve.

        Returns:
            Value of the AUPR metric
        """
        prec: torch.Tensor
        rec: torch.Tensor

        prec, rec = self._compute()
        return auc(rec, prec, reorder=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Need to flatten new values as PrecicionRecallCurve expects them in this format for binary classification.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prec/rec value pairs.

        Returns:
            Tuple containing Tensors for rec and prec
        """
        prec: torch.Tensor
        rec: torch.Tensor
        prec, rec, _ = super().compute()
        return (prec, rec)

class OptimalF1(Metric):
    full_state_update: bool = False
    def __init__(self, num_classes: int, **kwargs) -> None:
        msg = (
            "OptimalF1 metric is deprecated and will be removed in a future release. The optimal F1 score for "
            "Anomalib predictions can be obtained by computing the adaptive threshold with the "
            "AnomalyScoreThreshold metric and setting the computed threshold value in TorchMetrics F1Score metric."
        )
        super().__init__(**kwargs)

        self.precision_recall_curve = PrecisionRecallCurve(num_classes=num_classes)

        self.threshold: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> None:
        """Update the precision-recall curve metric."""
        del args, kwargs  # These variables are not used.

        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the value of the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        precision, recall, thresholds = self.precision_recall_curve.compute()
        f1_score_0 = (2 * precision[0] * recall[0]) / (precision[0] + recall[0] + 1e-10)
        f1_score_1 = (2 * precision[1] * recall[1]) / (precision[1] + recall[1] + 1e-10)
        self.threshold = thresholds[1][torch.argmax(f1_score_1)]
        return 0.5 * (torch.max(f1_score_0) + torch.max(f1_score_1))

    def reset(self) -> None:
        """Reset the metric."""
        self.precision_recall_curve.reset()
import torch
from torchmetrics import ROC
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

class PRO:
    pass
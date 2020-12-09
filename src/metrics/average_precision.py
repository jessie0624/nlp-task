'''
@description: 平均Precision@k
'''

import numpy as np
from src.base import BaseMetric
from src.metrics.precision import Precision

class AveragePrecision(BaseMetric):
    ALIAS = ['average_precision', 'ap']

    def __init__(self, threshold: float=0.):
        self._threshold = threshold

    def __repr__(self) -> str:
        return f"{self.ALIAS[0]}({self._threshold})"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision_metrics = [Precision(k+1) for k in range(y_pred.shape[1])]
        out = [metric(y_true, y_pred) for metric in precision_metrics]
        if not out:
            return 0.
        return np.mean(out).item()
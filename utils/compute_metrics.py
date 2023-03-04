from torch import Tensor, LongTensor, max
from typing import Dict
from sklearn.metrics import accuracy_score

def compute_metrics(
    outputs: Tensor,
    labels: LongTensor,
) -> Dict[str, float]:\

    metrics = {}

    outputs = outputs.cpu()
    labels = labels.cpu()
    _, pred = max(outputs.data, 1)

    y_true = labels
    y_pred = pred

    
    # accuracy
    accuracy = accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )

    # Optional add metrics

    metrics["accuracy"] = accuracy
    return metrics

if __name__ == '__main__':
    pass
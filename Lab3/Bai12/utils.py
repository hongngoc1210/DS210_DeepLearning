import torch
from sklearn.metrics import f1_score
import numpy as np

def calculate_f1(preds, labels):
    """
    preds: tensor shape (N, C) logits or probabilities
    labels: tensor shape (N,)
    returns: dict with f1_weighted, f1_per_class, accuracy, predictions, true_labels
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)

    # ensure on cpu numpy
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()

    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1_per_class = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    accuracy = (pred_labels == true_labels).mean()

    return {
        'f1_weighted': float(f1),
        'f1_per_class': [float(x) for x in f1_per_class],
        'accuracy': float(accuracy),
        'predictions': pred_labels,
        'true_labels': true_labels
    }

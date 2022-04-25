"""Compute f1 score between predict ids and label ids."""
from sklearn.metrics import classification_report


def compute_metrics(preds, labels):
    """
    Compute weighted F1.
    Args:
        preds: pred id list
        labels: gold id list

    Returns:
        res: result dict
    """
    assert len(preds) == len(labels)
    report = classification_report(labels, preds, output_dict=True)
    avg_precision = report["weighted avg"]["precision"]
    avg_recall = report["weighted avg"]["recall"]
    avg_f1 = report["weighted avg"]["f1-score"]

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }

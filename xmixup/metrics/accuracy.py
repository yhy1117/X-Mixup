"""Compute accuracy between predict ids and label ids."""


def compute_metrics(preds, labels):
    """
        Compute accuracy.
        Args:
            preds: pred id list
            labels: gold id list

        Returns:
            result dict
        """
    assert len(preds) == len(labels)
    acc = (preds == labels).mean()

    return {"acc": acc}

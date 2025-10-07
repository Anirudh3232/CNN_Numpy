import numpy as np

def one_hot(y, num_classes):
    """Return a one-hot vector for class index y (int)."""
    v = np.zeros(num_classes, dtype=float)
    v[y] = 1.0
    return v

def accuracy(preds, labels):
    """Compute accuracy for integer labels."""
    preds = np.array(preds, dtype=int)
    labels = np.array(labels, dtype=int)
    return float(np.mean(preds == labels))

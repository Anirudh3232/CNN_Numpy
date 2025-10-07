"""
Tiny synthetic dataset (offline, no downloads):
- Class 0: vertical line somewhere (x column set to 1s)
- Class 1: horizontal line somewhere (y row set to 1s)
Images are 28x28 grayscale (float32 in [0,1]).
"""

import numpy as np

def get_synthetic_lines(n_per_class=200, seed=42):
    rng = np.random.default_rng(seed)
    H, W = 28, 28
    X = []
    y = []
    # Class 0: vertical lines
    for _ in range(n_per_class):
        img = np.zeros((H, W), dtype=np.float32)
        col = rng.integers(low=6, high=W-6)  # avoid exact edges
        img[:, col] = 1.0
        # add a little noise
        img += 0.05 * rng.random((H, W))
        img = np.clip(img, 0.0, 1.0)
        X.append(img)
        y.append(0)

    # Class 1: horizontal lines
    for _ in range(n_per_class):
        img = np.zeros((H, W), dtype=np.float32)
        row = rng.integers(low=6, high=H-6)
        img[row, :] = 1.0
        img += 0.05 * rng.random((H, W))
        img = np.clip(img, 0.0, 1.0)
        X.append(img)
        y.append(1)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=int)

    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

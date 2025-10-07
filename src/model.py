import numpy as np

# --------------------------
# Layers
# --------------------------

class Conv3x3:
    """
    A simple 2D convolution with F filters of size 3x3, stride 1, no padding.
    Input:  (H, W)
    Output: (F, H-2, W-2)
    """
    def __init__(self, num_filters, weight_scale=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.num_filters = num_filters
        self.filters = weight_scale * rng.standard_normal((num_filters, 3, 3))
        self.last_input = None

    def iterate_regions(self, image):
        H, W = image.shape
        for i in range(H - 2):
            for j in range(W - 2):
                region = image[i:i+3, j:j+3]
                yield region, i, j

    def forward(self, x):
        self.last_input = x  # (H, W)
        H, W = x.shape
        out = np.zeros((self.num_filters, H-2, W-2), dtype=float)
        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(x):
                out[f, i, j] = np.sum(region * self.filters[f])
        return out

    def backward(self, d_out, lr):
        """
        d_out: gradient of loss wrt output, shape (F, H-2, W-2)
        returns: gradient wrt input image, shape (H, W)
        Updates self.filters using SGD.
        """
        x = self.last_input
        H, W = x.shape
        d_filters = np.zeros_like(self.filters)
        d_x = np.zeros_like(x)

        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(x):
                # scalar gradient for this output location
                grad = d_out[f, i, j]
                d_filters[f] += grad * region
                # distribute gradient back to the input region
                d_x[i:i+3, j:j+3] += grad * self.filters[f]

        # SGD update
        self.filters -= lr * d_filters
        return d_x


class ReLU:
    """Elementwise ReLU activation."""
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0).astype(float)
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class MaxPool2:
    """
    2x2 max pooling with stride=2.
    Input:  (F, H, W)
    Output: (F, H//2, W//2)
    """
    def __init__(self):
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        F, H, W = x.shape
        out = np.zeros((F, H//2, W//2), dtype=float)
        for f in range(F):
            for i in range(H // 2):
                for j in range(W // 2):
                    region = x[f, i*2:i*2+2, j*2:j*2+2]
                    out[f, i, j] = np.max(region)
        return out

    def backward(self, d_out):
        x = self.last_input
        F, H, W = x.shape
        d_x = np.zeros_like(x)

        for f in range(F):
            for i in range(H // 2):
                for j in range(W // 2):
                    region = x[f, i*2:i*2+2, j*2:j*2+2]
                    (r, c) = np.unravel_index(np.argmax(region), (2, 2))
                    d_x[f, i*2 + r, j*2 + c] = d_out[f, i, j]
        return d_x


class Flatten:
    """Flattens (F, H, W) to (F*H*W,)"""
    def __init__(self):
        self.last_shape = None

    def forward(self, x):
        self.last_shape = x.shape
        return x.reshape(-1)

    def backward(self, d_out):
        return d_out.reshape(self.last_shape)


class Dense:
    """Fully connected layer y = Wx + b"""
    def __init__(self, in_features, out_features, weight_scale=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.W = weight_scale * rng.standard_normal((out_features, in_features))
        self.b = np.zeros(out_features, dtype=float)
        self.last_x = None

    def forward(self, x):
        self.last_x = x  # shape (in_features,)
        return self.W @ x + self.b

    def backward(self, d_out, lr):
        # d_out shape: (out_features,)
        dW = np.outer(d_out, self.last_x)
        db = d_out
        dx = self.W.T @ d_out
        # SGD step
        self.W -= lr * dW
        self.b -= lr * db
        return dx


# --------------------------
# Loss
# --------------------------

class SoftmaxCrossEntropy:
    """
    Combines softmax and cross-entropy for numerical stability.
    Forward returns (probs, loss_scalar).
    Backward returns d_logits.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.last_logits = None
        self.last_probs = None
        self.last_label = None  # integer class

    def forward(self, logits, y_int):
        # logits shape: (C,)
        self.last_logits = logits.copy()
        self.last_label = int(y_int)

        # stable softmax
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z)
        self.last_probs = probs

        # cross-entropy loss
        loss = -np.log(probs[self.last_label] + 1e-12)
        return probs, loss

    def backward(self):
        # gradient wrt logits: p - one_hot(y)
        grad = self.last_probs.copy()
        grad[self.last_label] -= 1.0
        return grad

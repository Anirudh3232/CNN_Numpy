"""
Beginner-friendly training script:
CNN(Conv3x3 -> ReLU -> MaxPool2 -> Flatten -> Dense) + SoftmaxCrossEntropy
on a tiny synthetic dataset (vertical vs horizontal line).

Run:
  python main.py

Expected behavior: loss drops, accuracy rises above chance.
"""

import numpy as np
from data_loader import get_synthetic_lines
from model import Conv3x3, ReLU, MaxPool2, Flatten, Dense, SoftmaxCrossEntropy
from utils import accuracy

# 1) Data
X, y = get_synthetic_lines(n_per_class=300, seed=1)  # 600 samples, 28x28
num_classes = 2

# Simple split
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]

# 2) Model
conv   = Conv3x3(num_filters=8, weight_scale=0.1, seed=0)  # (H,W)->(8,H-2,W-2) => (8,26,26)
relu   = ReLU()
pool   = MaxPool2()                                        # (8,26,26)->(8,13,13)
flat   = Flatten()
dense  = Dense(in_features=8*13*13, out_features=num_classes, weight_scale=0.01, seed=0)
lossfn = SoftmaxCrossEntropy(num_classes=num_classes)

lr = 0.01
epochs = 5

def forward_pass(x_img):
    # x_img: (28,28)
    out = conv.forward(x_img)
    out = relu.forward(out)
    out = pool.forward(out)
    out = flat.forward(out)
    logits = dense.forward(out)
    return logits

def backward_pass(d_logits):
    # d_logits: gradient wrt final layer logits
    d = dense.backward(d_logits, lr=lr)
    d = flat.backward(d)
    d = pool.backward(d)
    d = relu.backward(d)
    d = conv.backward(d, lr=lr)
    return d

def predict(x_img):
    logits = forward_pass(x_img)
    probs, _ = lossfn.forward(logits, y_int=0)  # y not used for probs; dummy 0
    return int(np.argmax(probs))

# 3) Training loop (SGD, single example)
for ep in range(1, epochs+1):
    # shuffle each epoch
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    running_loss = 0.0
    preds = []

    for i in range(len(X_train)):
        x_img = X_train[i]
        y_int = int(y_train[i])

        # Forward
        logits = forward_pass(x_img)
        probs, loss = lossfn.forward(logits, y_int)
        running_loss += loss

        # Prediction for train accuracy snapshot
        preds.append(int(np.argmax(probs)))

        # Backward
        d_logits = lossfn.backward()          # dL/dlogits
        backward_pass(d_logits)               # updates parameters via SGD inside

    train_acc = accuracy(preds, y_train)
    # Evaluate on test set
    test_preds = [predict(img) for img in X_test]
    test_acc = accuracy(test_preds, y_test)

    print(f"Epoch {ep:02d} | loss={running_loss/len(X_train):.4f} | "
          f"train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")

print("\\nTraining complete.")
print("Try tweaking: num_filters, lr, epochs. Replace dataset with MNIST later if desired.")

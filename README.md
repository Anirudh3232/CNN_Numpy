## CNN in NumPy

A beginner-friendly Convolutional Neural Network implemented from scratch in NumPy.
The goal is learning: every layer has a short forward() and backward() so you can follow how CNNs really work. 

Pipeline: Conv3×3 → ReLU → MaxPool(2×2) → Flatten → Dense → Softmax Cross-Entropy (with SGD updates)

Why this repo?

Pure NumPy: no PyTorch/TensorFlow—perfect for understanding the math.

Small codebase: read it in an hour, modify in minutes.

Offline training: includes a tiny synthetic dataset (vertical vs. horizontal lines).

Project structure
CNN_Numpy/
 README.md
 requirements.txt     # numpy (and pytest if you add tests)
src/
    main.py           # entry point: trains & evaluates the model
     model.py          # layers + loss (forward/backward) + SGD updates
    data_loader.py    # synthetic 28×28 dataset (two classes + noise)
   utils.py          # accuracy, seed helpers

Quickstart
Local (CPU is enough)
git clone https://github.com/Anirudh3232/CNN_Numpy.git
cd CNN_Numpy
pip install -r requirements.txt
python src/main.py


Expected output (numbers will vary):

Epoch 01 | loss=0.69.. | train_acc=0.85 | test_acc=0.84
...

Google Colab
!git clone https://github.com/Anirudh3232/CNN_Numpy.git
%cd CNN_Numpy
!pip install -r requirements.txt
!python src/main.py

How the code works (short tour)

data_loader.py – builds a simple dataset so training works offline:

Class 0: a vertical line in a 28×28 image

Class 1: a horizontal line

Adds small noise so the model must actually learn, not memorize.

model.py – minimal CNN building blocks:

Conv3x3: slides 3×3 filters across the image (feature detection)

ReLU: keeps positives, zeros negatives (non-linearity)

MaxPool2: downsample by 2×2 windows (keep strongest activations)

Flatten: reshape (F,H,W) → (F·H·W,)

Dense: fully-connected classifier head

SoftmaxCrossEntropy: stable softmax + cross-entropy loss

Each layer has forward() and backward(); weights update via simple SGD.

utils.py – small helpers:

accuracy(preds, labels) and set_seed(seed).

main.py – training loop:

load data & split train/test

define layers and loss

for each epoch: forward - loss - backward - update

print loss and accuracy for quick feedback.

Tweak me (great first experiments)

Filters: in main.py set num_filters=8 - 16 or 32

Epochs: increase epochs to see steadier accuracy

Learning rate: try lr=0.005 or 0.02

Harder data: bump noise in data_loader.py from 0.05 - 0.10–0.15

Add a block: another Conv3x3 + ReLU + MaxPool2 before Flatten

Mini-batches: rewrite the loop to process N×H×W at once

Troubleshooting

Import errors: run from repo root (python src/main.py), not inside src/.

Colab resets: if the runtime restarts, re-run the clone cell.

No GPU use: this is NumPy-only by design; CPU is fine.

Roadmap (nice next steps)

Add an MNIST loader (download once; cache to data/)

Plot loss/accuracy curves

Weight decay / momentum

Unit tests for shape and gradient sanity checks

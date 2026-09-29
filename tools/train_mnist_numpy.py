#! /usr/bin/env python3
import gzip
import numpy as np
import struct
from pathlib import Path
import urllib.request

DATA_DIR  = Path(__file__).parent.parent / "data" / "mnist"
MODEL_DIR = Path(__file__).parent.parent / "data"/ "models"

MIRROR = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}

def download_data():
    print(f"Downloading MNIST data to {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
 
    for file in FILES.values():
        dst_file = DATA_DIR / file
        if dst_file.exists():
            print(f"File {dst_file} already exists")
            continue
        for site in MIRROR:
            url = f"{site}{file}"
            print(f"Downloading {url} to {DATA_DIR / file}")
            urllib.request.urlretrieve(url, dst_file)
            break
        else:
            print(f"Failed to download {file} from any mirror")
            raise RuntimeError(f"Failed to download {file} from any mirror")

def read_idx_label(file):
    with gzip.open(file, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")
        buf = f.read()
    labels = np.frombuffer(buf, dtype=np.uint8)
    if labels.shape[0] != n:
        raise ValueError(f"Invalid number of labels: {labels.shape[0]} != {n}")
    return labels

def read_idx_image(file):
    with gzip.open(file, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic {magic} in {file}")
        buf = f.read()
    pixels = np.frombuffer(buf, dtype=np.uint8)
    if pixels.size != n * rows * cols:
        raise ValueError(f"size mismatch in {file}")
    return pixels.reshape(n, rows , cols)

def load_data() :
    print(f"Loading MNIST data from {DATA_DIR}")
    train_x = read_idx_image(DATA_DIR / FILES["train_x"]).reshape(-1, 784).astype(np.float32) / 255.0
    train_y = read_idx_label(DATA_DIR / FILES["train_y"]).astype(np.int64)
    test_x = read_idx_image(DATA_DIR / FILES["test_x"]).reshape(-1, 784).astype(np.float32) / 255.0
    test_y = read_idx_label(DATA_DIR / FILES["test_y"]).astype(np.int64)
    return train_x, train_y, test_x, test_y


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

class MLP:
    def __init__(self, d_in = 784, d_hidden = 128, d_out = 10, seed = 0 , batch_size = 128, learning_rate = 0.1, epochs = 12):
        np.random.seed(seed)
        # He init; W shape (out, in) like nn.Linear
        self.W1 = (np.random.randn(d_hidden, d_in) * np.sqrt(2 / d_in)).astype(np.float32)
        self.b1 = np.zeros(d_hidden, dtype=np.float32)
        self.W2 = (np.random.randn(d_out, d_hidden) * np.sqrt(2 / d_hidden)).astype(np.float32)
        self.b2 = np.zeros(d_out, dtype=np.float32)
        self.relu = lambda x: np.maximum(0, x)
        self.softmax = softmax

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def forward(self, x):
        z1 = x @ self.W1.T + self.b1
        h1 = self.relu(z1)
        z2 = h1 @ self.W2.T + self.b2
        p = self.softmax(z2)
        return z1, h1, z2, p

    def backward(self, x_batch, y_batch, z1, h1, p):
        """W is (out, in); forward uses x @ W.T (same as nn.Linear)."""
        batch_size = x_batch.shape[0]
        y_onehot = np.zeros_like(p)
        y_onehot[np.arange(batch_size), y_batch] = 1.0

        delta = (p - y_onehot) / batch_size

        dW2 = delta.T @ h1
        db2 = delta.sum(axis=0)

        delta = (delta @ self.W2) * (z1 > 0)

        dW1 = delta.T @ x_batch
        db1 = delta.sum(axis=0)
        return dW1, db1, dW2, db2

    def predict(self, x):
        return np.argmax(self.forward(x)[2], axis=1)

    def train(self, x, y):
        for epoch in range(self.epochs):
            loss = 0.0
            for i in range(0, x.shape[0], self.batch_size):
                # prepare batch for training batch size = 128
                x_batch = x[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                # forward pass
                z1, h1, z2 ,p = self.forward(x_batch)
                batch_size = x_batch.shape[0]
                loss += -np.log(
                    p[np.arange(batch_size), y_batch] + 1e-9
                ).sum()
                # backward pass
                dW1, db1, dW2, db2 = self.backward(x_batch, y_batch, z1, h1, p)
                # update parameters
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
            print(f"Epoch {epoch+1}, Loss: {loss / x.shape[0]}")
        return self
    def evaluate(self, x, y):
        return np.mean(self.predict(x) == y)


if __name__ == "__main__":
    # step1 download mnist data
    download_data()
    # step2 load mnist data
    train_x, train_y, test_x, test_y = load_data()
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    # step3 train model
    model = MLP()
    model.train(train_x, train_y)
    # step4 evaluate model
    accuracy = model.evaluate(test_x, test_y)
    print(f"Test accuracy: {accuracy}")

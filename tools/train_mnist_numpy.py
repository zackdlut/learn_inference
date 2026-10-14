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
    def __init__(self, d_in = 784, d_hidden = 128, d_out = 10, seed = 0 , batch_size = 100, learning_rate = 0.1, epochs = 12):
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

        self.MAGIC = b"INFR"
        self.VERSION = 1
        self.DTYPE_F32 = 0

    def forward(self, x):
        z1 = x @ self.W1.T + self.b1
        h1 = self.relu(z1)
        z2 = h1 @ self.W2.T + self.b2
        prob = self.softmax(z2)
        return z1, h1, z2, prob

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

    def train(
        self,
        x,
        y,
        *,
        test_x=None,
        test_y=None,
        train_eval_x=None,
        train_eval_y=None,
        verbose=True,
        eval_every: int = 1,
    ):
        """Train and record per-epoch metrics in ``self.history``."""
        self.history = []
        n = x.shape[0]
        eval_every = max(1, eval_every)
        for epoch in range(self.epochs):
            loss = 0.0
            steps = 0
            for i in range(0, n, self.batch_size):
                x_batch = x[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]
                z1, h1, z2, prob = self.forward(x_batch)
                batch_size = x_batch.shape[0]
                loss += -np.log(
                    prob[np.arange(batch_size), y_batch] + 1e-9
                ).sum()
                dW1, db1, dW2, db2 = self.backward(x_batch, y_batch, z1, h1, prob)
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
                steps += 1
            row = {
                "epoch": epoch + 1,
                "train_loss": float(loss / n),
                "steps_per_epoch": steps,
            }
            do_eval = (epoch + 1) % eval_every == 0 or epoch + 1 == self.epochs
            if do_eval:
                if train_eval_x is not None and train_eval_y is not None:
                    row["train_acc"] = float(self.evaluate(train_eval_x, train_eval_y))
                if test_x is not None and test_y is not None:
                    row["test_acc"] = float(self.evaluate(test_x, test_y))
            self.history.append(row)
            if verbose:
                msg = f"Epoch {epoch + 1}, loss={row['train_loss']:.4f}"
                if "train_acc" in row:
                    msg += f", train_acc={row['train_acc']:.4f}"
                if "test_acc" in row:
                    msg += f", test_acc={row['test_acc']:.4f}"
                print(msg)
        return self
    def evaluate(self, x, y):
        return np.mean(self.predict(x) == y)
    
    def save_model(self, path):
        self.save(path, {"fc1.weight": self.W1, "fc1.bias": self.b1, "fc2.weight": self.W2, "fc2.bias": self.b2})
        print(f"Model saved to {path}")

    # ---------------------------------------------------------------------------
    # 模型文件格式
    #
    #   magic  "INFR"          4 字节，认文件用
    #   version                uint32，以后改格式靠它兼容老文件
    #   num_tensors            uint32
    #   重复 num_tensors 次:
    #       name_len  uint32
    #       name      UTF-8
    #       dtype     uint32   (0 = f32)
    #       ndim      uint32
    #       dims      int64 × ndim
    #       offset    uint64   ← 数据在「数据区」内的字节偏移
    #       nbytes    uint64
    #   data_start             uint64，数据区起始位置（从文件头算）
    #   [数据区]                所有张量的原始字节，紧密排列
    #
    # 为什么元信息和数据要分开放？
    #   这样加载时可以先只读元信息（很小），需要哪个张量再去读哪一段；
    #   到阶段 7 还能直接 mmap 数据区，做到「零拷贝加载」—— 几百 MB 的模型
    #   瞬间就绪，多个进程还能共享同一份物理内存。
    # ---------------------------------------------------------------------------
    def save(self, path, tensors):
        # create parent directory if not exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # prepare body, meta, cursor
        body, meta , cursor = [], [], 0
        for name, tensor in tensors.items():
            tensor_contiguous = np.ascontiguousarray(tensor, dtype=np.float32)
            ternsor_raw = tensor_contiguous.tobytes()
            meta.append((name, tensor_contiguous.shape, cursor, len(ternsor_raw)))
            body.append(ternsor_raw)
            cursor += len(ternsor_raw)

        # pack header 
        # header = magic + version + meta length + meta + data_start
        header = bytearray()
        header += self.MAGIC
        header += struct.pack("<I", self.VERSION)
        #pack meta length + meta
        header += struct.pack("<I", len(meta))
        for name, shape, offset, length in meta:
            # pack name length + name
            name_bytes = name.encode("utf-8")
            header += struct.pack("<I", len(name_bytes))
            header += name_bytes
            # pack data type
            header += struct.pack("<I", self.DTYPE_F32)
            # pack shape length + shape
            header += struct.pack("<I", len(shape))
            for dim in shape:
                header += struct.pack("<q", dim)
            # pack offset + length
            header += struct.pack("<Q", offset)
            header += struct.pack("<Q", length)
        # data_start 字段本身占 8 字节，要算进去
        data_start = len(header) + 8
        header += struct.pack("<Q", data_start)
        # pack body
        with open(path, "wb") as f:
            f.write(header)
            for data in body:
                f.write(data)
        print(f"Model saved to {path}")

    def save_activation(self, path, x):
        # save input, z1, h1, z2, prob
        one = x[:1]
        z1, h1, z2, prob = self.forward(one)
        self.save(path, {"input": one, "z1": z1, "h1": h1, "z2": z2, "prob": prob})
        print(f"Activation saved to {path}")

    def save_samples(self, path, x, y , n = 100):
        # save images and labels
        from_golden = {}
        from_golden["images"] = x[:n]
        from_golden["labels"] = y[:n].astype(np.float32)
        self.save(path, from_golden)
        print(f"Samples saved to {path}")

if __name__ == "__main__":
    # step1 download mnist data
    download_data()
    # step2 load mnist data
    train_x, train_y, test_x, test_y = load_data()
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    # step3 train model
    model = MLP()
    model.train(
        train_x,
        train_y,
        test_x=test_x,
        test_y=test_y,
        train_eval_x=train_x[:5000],
        train_eval_y=train_y[:5000],
    )
    print(f"Test accuracy: {model.evaluate(test_x, test_y)}")
    # step5 save model
    model.save_model(MODEL_DIR / "mnist_mlp.inference")
    # step6 save activation
    model.save_activation(MODEL_DIR / "mnist_activation.inference", test_x)
    # step7 save samples
    model.save_samples(MODEL_DIR / "mnist_samples.inference", test_x, test_y, n=200)
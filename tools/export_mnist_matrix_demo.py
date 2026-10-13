#!/usr/bin/env python3
"""Export one mini-batch step (B=100) with all tensors for matrix demo HTML."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from train_mnist_numpy import load_data, MLP  # noqa: E402


def round_arr(a: np.ndarray, digits: int = 6) -> list:
    return np.round(a.astype(np.float64), digits).tolist()


def main() -> None:
    train_x, train_y, _, _ = load_data()
    B = 100
    m = MLP(seed=0, batch_size=B, learning_rate=0.1, epochs=1)
    x_batch = train_x[:B].copy()
    y_batch = train_y[:B].copy()

    W1_0, b1_0 = m.W1.copy(), m.b1.copy()
    W2_0, b2_0 = m.W2.copy(), m.b2.copy()

    z1, h1, z2, p = m.forward(x_batch)
    batch_size = x_batch.shape[0]
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(batch_size), y_batch] = 1.0
    delta = (p - y_onehot) / batch_size
    dW2 = delta.T @ h1
    db2 = delta.sum(axis=0)
    delta_h = delta @ m.W2
    delta_z1 = delta_h * (z1 > 0)
    dW1 = delta_z1.T @ x_batch
    db1 = delta_z1.sum(axis=0)

    lr = m.learning_rate
    W1_1 = W1_0 - lr * dW1
    b1_1 = b1_0 - lr * db1
    W2_1 = W2_0 - lr * dW2
    b2_1 = b2_0 - lr * db2

    loss_batch = float(
        -np.log(p[np.arange(batch_size), y_batch] + 1e-9).sum()
    )

    out = {
        "meta": {
            "batch_size": B,
            "batch_index": 0,
            "epoch": 1,
            "step_in_epoch": 0,
            "learning_rate": lr,
            "loss_batch_sum": loss_batch,
            "steps_per_epoch": train_x.shape[0] // B,
            "note": "第一个 mini-batch：train_x[0:100], train_y[0:100]",
        },
        "labels": y_batch.tolist(),
        "shapes": {
            "x": list(x_batch.shape),
            "W1": list(W1_0.shape),
            "b1": list(b1_0.shape),
            "z1": list(z1.shape),
            "h1": list(h1.shape),
            "W2": list(W2_0.shape),
            "b2": list(b2_0.shape),
            "z2": list(z2.shape),
            "p": list(p.shape),
            "delta": list(delta.shape),
            "delta_z1": list(delta_z1.shape),
            "dW1": list(dW1.shape),
            "db1": list(db1.shape),
            "dW2": list(dW2.shape),
            "db2": list(db2.shape),
        },
        "before": {
            "W1": round_arr(W1_0),
            "b1": round_arr(b1_0),
            "W2": round_arr(W2_0),
            "b2": round_arr(b2_0),
        },
        "after": {
            "W1": round_arr(W1_1),
            "b1": round_arr(b1_1),
            "W2": round_arr(W2_1),
            "b2": round_arr(b2_1),
        },
        "batch": {
            "x": round_arr(x_batch),
            "z1": round_arr(z1),
            "h1": round_arr(h1),
            "z2": round_arr(z2),
            "p": round_arr(p),
            "delta": round_arr(delta),
            "delta_z1": round_arr(delta_z1),
        },
        "grads": {
            "dW1": round_arr(dW1),
            "db1": round_arr(db1),
            "dW2": round_arr(dW2),
            "db2": round_arr(db2),
        },
    }

    dst = ROOT / "docs" / "train_mnist_matrix_demo_data.json"
    raw = json.dumps(out, separators=(",", ":")).encode("utf-8")
    dst.write_bytes(raw)
    dst_gz = ROOT / "docs" / "train_mnist_matrix_demo_data.json.gz"
    dst_gz.write_bytes(gzip.compress(raw, compresslevel=6))
    mb = dst.stat().st_size / (1024 * 1024)
    mb_gz = dst_gz.stat().st_size / (1024 * 1024)
    print(f"Wrote {dst} ({mb:.2f} MiB)")
    print(f"Wrote {dst_gz} ({mb_gz:.2f} MiB)")


if __name__ == "__main__":
    main()

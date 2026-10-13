#!/usr/bin/env python3
"""Compare batch_size / learning_rate for train_mnist_numpy.MLP and plot curves."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_mnist_numpy import MLP, load_data

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures"


def run_experiment(
    train_x,
    train_y,
    test_x,
    test_y,
    train_eval_x,
    train_eval_y,
    *,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    verbose: bool,
    eval_every: int,
) -> dict:
    model = MLP(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )
    t0 = time.perf_counter()
    model.train(
        train_x,
        train_y,
        test_x=test_x,
        test_y=test_y,
        train_eval_x=train_eval_x,
        train_eval_y=train_eval_y,
        verbose=verbose,
        eval_every=eval_every,
    )
    elapsed = time.perf_counter() - t0
    hist = model.history
    # fill missing accuracy points for plotting (copy last known)
    last_ta = last_va = np.nan
    for row in hist:
        if "train_acc" in row:
            last_ta = row["train_acc"]
        elif not np.isnan(last_ta):
            row["train_acc"] = last_ta
        if "test_acc" in row:
            last_va = row["test_acc"]
        elif not np.isnan(last_va):
            row["test_acc"] = last_va
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history": hist,
        "final_test_acc": hist[-1].get("test_acc", float("nan")),
        "final_train_loss": hist[-1]["train_loss"],
        "wall_time_s": elapsed,
        "total_steps": sum(h["steps_per_epoch"] for h in hist),
    }


def _epochs_axis(history: list[dict]) -> np.ndarray:
    return np.array([h["epoch"] for h in history], dtype=int)


def plot_batch_size_study(results: list[dict], out_path: Path, fixed_lr: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Batch size comparison (lr={fixed_lr}, same seed)", fontsize=13)

    ax_loss, ax_test, ax_train, ax_bar = axes.ravel()

    for r in sorted(results, key=lambda x: x["batch_size"]):
        h = r["history"]
        ep = _epochs_axis(h)
        label = f"batch={r['batch_size']} ({r['total_steps']} steps)"
        ax_loss.plot(ep, [x["train_loss"] for x in h], marker="o", markersize=3, label=label)
        ax_test.plot(ep, [x["test_acc"] for x in h], marker="o", markersize=3, label=label)
        ax_train.plot(ep, [x["train_acc"] for x in h], marker="o", markersize=3, label=label)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss (mean CE)")
    ax_loss.set_title("Loss vs epoch")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_test.set_xlabel("Epoch")
    ax_test.set_ylabel("Test accuracy")
    ax_test.set_title("Test accuracy vs epoch")
    ax_test.grid(True, alpha=0.3)
    ax_test.legend(fontsize=8)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train accuracy (subset)")
    ax_train.set_title("Train accuracy vs epoch")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(fontsize=8)

    names = [str(r["batch_size"]) for r in results]
    finals = [r["final_test_acc"] for r in results]
    times = [r["wall_time_s"] for r in results]
    xpos = np.arange(len(names))
    bars = ax_bar.bar(xpos, finals, color="steelblue", alpha=0.85)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(names)
    ax_bar.set_xlabel("batch_size")
    ax_bar.set_ylabel("Final test accuracy")
    ax_bar.set_title("Final test acc & wall time")
    ax_bar.set_ylim(0, 1.0)
    for bar, t in zip(bars, times):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{t:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lr_study(results: list[dict], out_path: Path, fixed_batch: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Learning rate comparison (batch_size={fixed_batch}, same seed)", fontsize=13)

    ax_loss, ax_test, ax_train, ax_bar = axes.ravel()

    for r in sorted(results, key=lambda x: x["learning_rate"]):
        h = r["history"]
        ep = _epochs_axis(h)
        label = f"lr={r['learning_rate']}"
        ax_loss.plot(ep, [x["train_loss"] for x in h], marker="o", markersize=3, label=label)
        ax_test.plot(ep, [x["test_acc"] for x in h], marker="o", markersize=3, label=label)
        ax_train.plot(ep, [x["train_acc"] for x in h], marker="o", markersize=3, label=label)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss (mean CE)")
    ax_loss.set_title("Loss vs epoch")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_test.set_xlabel("Epoch")
    ax_test.set_ylabel("Test accuracy")
    ax_test.set_title("Test accuracy vs epoch")
    ax_test.grid(True, alpha=0.3)
    ax_test.legend(fontsize=8)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train accuracy (subset)")
    ax_train.set_title("Train accuracy vs epoch")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(fontsize=8)

    names = [str(r["learning_rate"]) for r in results]
    finals = [r["final_test_acc"] for r in results]
    xpos = np.arange(len(names))
    ax_bar.bar(xpos, finals, color="coral", alpha=0.85)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(names)
    ax_bar.set_xlabel("learning_rate")
    ax_bar.set_ylabel("Final test accuracy")
    ax_bar.set_title("Final test accuracy")
    ax_bar.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_table(batch_results: list[dict], lr_results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    rows = []
    for r in batch_results:
        rows.append(
            [
                "batch_size",
                str(r["batch_size"]),
                f"{r['learning_rate']}",
                f"{r['final_train_loss']:.4f}",
                f"{r['final_test_acc']:.4f}",
                str(r["total_steps"]),
                f"{r['wall_time_s']:.1f}",
            ]
        )
    for r in lr_results:
        rows.append(
            [
                "lr",
                str(r["batch_size"]),
                f"{r['learning_rate']}",
                f"{r['final_train_loss']:.4f}",
                f"{r['final_test_acc']:.4f}",
                str(r["total_steps"]),
                f"{r['wall_time_s']:.1f}",
            ]
        )
    cols = ["study", "batch", "lr", "final loss", "final test acc", "total steps", "time(s)"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.scale(1, 1.4)
    ax.set_title("Experiment summary", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MNIST MLP training comparisons")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[64, 100, 128, 256],
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[0.02, 0.1, 0.5],
    )
    parser.add_argument("--fixed-lr", type=float, default=0.1, help="lr for batch-size study")
    parser.add_argument(
        "--fixed-batch",
        type=int,
        default=128,
        help="batch_size for lr study",
    )
    parser.add_argument("--train-eval-size", type=int, default=5000)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="run test/train accuracy every N epochs (1=every epoch)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="shorter run: epochs=5, batch_sizes=100 128, lrs=0.05 0.1 0.5",
    )
    args = parser.parse_args()
    if args.fast:
        args.epochs = 5
        args.batch_sizes = [100, 128]
        args.lrs = [0.05, 0.1, 0.5]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = load_data()
    n_eval = min(args.train_eval_size, train_x.shape[0])
    train_eval_x = train_x[:n_eval]
    train_eval_y = train_y[:n_eval]

    verbose = not args.quiet
    batch_results = []
    for bs in args.batch_sizes:
        print(f"=== batch_size={bs}, lr={args.fixed_lr} ===")
        batch_results.append(
            run_experiment(
                train_x,
                train_y,
                test_x,
                test_y,
                train_eval_x,
                train_eval_y,
                batch_size=bs,
                learning_rate=args.fixed_lr,
                epochs=args.epochs,
                seed=args.seed,
                verbose=verbose,
                eval_every=args.eval_every,
            )
        )

    lr_results = []
    for lr in args.lrs:
        print(f"=== batch_size={args.fixed_batch}, lr={lr} ===")
        lr_results.append(
            run_experiment(
                train_x,
                train_y,
                test_x,
                test_y,
                train_eval_x,
                train_eval_y,
                batch_size=args.fixed_batch,
                learning_rate=lr,
                epochs=args.epochs,
                seed=args.seed,
                verbose=verbose,
                eval_every=args.eval_every,
            )
        )

    batch_png = OUT_DIR / "mnist_batch_size_compare.png"
    lr_png = OUT_DIR / "mnist_lr_compare.png"
    table_png = OUT_DIR / "mnist_training_summary.png"

    plot_batch_size_study(batch_results, batch_png, args.fixed_lr)
    plot_lr_study(lr_results, lr_png, args.fixed_batch)
    plot_summary_table(batch_results, lr_results, table_png)

    summary = {
        "epochs": args.epochs,
        "seed": args.seed,
        "fixed_lr": args.fixed_lr,
        "fixed_batch": args.fixed_batch,
        "batch_size_study": batch_results,
        "lr_study": lr_results,
    }
    # history is JSON-serializable; strip for compact summary
    def slim(r):
        return {k: v for k, v in r.items() if k != "history"}

    summary_path = OUT_DIR / "mnist_training_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "batch_size_study": [slim(r) for r in batch_results],
                "lr_study": [slim(r) for r in lr_results],
                "batch_size_histories": {str(r["batch_size"]): r["history"] for r in batch_results},
                "lr_histories": {str(r["learning_rate"]): r["history"] for r in lr_results},
            },
            f,
            indent=2,
        )

    print(f"Wrote {batch_png}")
    print(f"Wrote {lr_png}")
    print(f"Wrote {table_png}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

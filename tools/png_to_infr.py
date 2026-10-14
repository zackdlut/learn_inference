#!/usr/bin/env python3
"""将 PNG 转为 28×28 灰度 MNIST 风格张量，写入 INFR（.inference）供 C++ ModelFile 读取。

预处理（默认 ``--mnist-style``）：白底合成 → 灰度 → 自动对比度 → 按墨迹裁切 →
缩放到约 20×20 并居中贴在 28×28 白底上 → **``1.0 - pixel/255``**。

与 ``train_mnist_numpy.load_data`` 一致：MNIST 原始 0=白底、255=黑字，除以 255 后
**0≈背景、1≈墨迹**。若只做 ``/255``，白底会变成 1.0，在 ``hello_mlp`` 的 ``draw()``
里会像「黑底白字」，模型输入也与训练分布相反。

手写字体/图标与 MNIST 差异很大时，模型仍可能判错；可用 ``--mnist-digit N`` 从测试集导出标准样本做联调。

默认每个 PNG 生成同名 .inference，内含张量 ``image``，形状 ``(1, 784)``。
可选 ``--label 3`` 额外写入 ``label``，形状 ``(1,)``，与 mnist_samples 一致。

示例::

    python3 tools/png_to_infr.py examples/3.png examples/4.png
    python3 tools/png_to_infr.py examples/3.png -o data/models/digit3.inference --label 3
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MAGIC = b"INFR"
VERSION = 1
DTYPE_F32 = 0


def save_infr(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """与 tools/train_mnist_numpy.py 中 MLP.save 相同的 INFR 布局。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body: list[bytes] = []
    meta: list[tuple[str, tuple[int, ...], int, int]] = []
    cursor = 0
    for name, tensor in tensors.items():
        contiguous = np.ascontiguousarray(tensor, dtype=np.float32)
        raw = contiguous.tobytes()
        meta.append((name, contiguous.shape, cursor, len(raw)))
        body.append(raw)
        cursor += len(raw)

    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<I", len(meta))
    for name, shape, offset, length in meta:
        name_bytes = name.encode("utf-8")
        header += struct.pack("<I", len(name_bytes))
        header += name_bytes
        header += struct.pack("<I", DTYPE_F32)
        header += struct.pack("<I", len(shape))
        for dim in shape:
            header += struct.pack("<q", int(dim))
        header += struct.pack("<Q", offset)
        header += struct.pack("<Q", length)
    data_start = len(header) + 8
    header += struct.pack("<Q", data_start)

    with path.open("wb") as f:
        f.write(header)
        for chunk in body:
            f.write(chunk)


def png_to_mnist_batch(
    path: Path,
    size: int = 28,
    *,
    mnist_style: bool = True,
    mnist_invert: bool = True,
    inner: int = 20,
    ink_threshold: int = 200,
) -> np.ndarray:
    """读取 PNG，返回 shape ``(1, size*size)`` 的 float32 向量（行优先展平）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    im = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(background, im).convert("L")

    if mnist_style:
        im = ImageOps.autocontrast(im, cutoff=0)
        arr = np.asarray(im)
        ink = arr < ink_threshold
        if ink.any():
            ys, xs = np.where(ink)
            im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        im = im.copy()
        im.thumbnail((inner, inner), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (size, size), 255)
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        im = canvas
    elif im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.LANCZOS)

    gray = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
    pixels = (1.0 - gray) if mnist_invert else gray
    expected = size * size
    if pixels.size != expected:
        raise ValueError(f"{path}: expected {expected} pixels, got {pixels.size}")
    return pixels.reshape(1, expected)


def mnist_test_digit_batch(digit: int) -> np.ndarray:
    """从 MNIST 测试集取该数字的第一张图，已是 (1, 784) float32 / 255。"""
    if not 0 <= digit <= 9:
        raise ValueError(f"digit must be 0..9, got {digit}")
    # 延迟导入，避免 --help 也拉数据
    from train_mnist_numpy import load_data

    _, _, test_x, test_y = load_data()
    indices = np.where(test_y == digit)[0]
    if indices.size == 0:
        raise ValueError(f"no test image for digit {digit}")
    return test_x[int(indices[0]) : int(indices[0]) + 1].copy()


def convert_one(
    png_path: Path,
    output_path: Path,
    *,
    label: int | None,
    tensor_name: str,
    mnist_style: bool,
    mnist_invert: bool,
) -> None:
    image = png_to_mnist_batch(
        png_path, mnist_style=mnist_style, mnist_invert=mnist_invert
    )
    tensors: dict[str, np.ndarray] = {tensor_name: image}
    if label is not None:
        if not 0 <= label <= 9:
            raise ValueError(f"label must be 0..9, got {label}")
        tensors["label"] = np.array([float(label)], dtype=np.float32)
    save_infr(output_path, tensors)
    print(f"Wrote {output_path}  ({tensor_name} {image.shape}", end="")
    if label is not None:
        print(f", label [{label}]", end="")
    print(")")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PNG → 28×28 灰度 INFR (.inference)")
    parser.add_argument("png", nargs="*", type=Path, help="输入 PNG（可多个）")
    parser.add_argument(
        "--mnist-digit",
        type=int,
        default=None,
        metavar="N",
        help="不从 PNG 读图，改用 MNIST 测试集数字 N（0–9），用于与 hello_mlp 联调",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出路径（仅当只指定一个 PNG 时可用；默认改为同名 .inference）",
    )
    parser.add_argument("--label", type=int, default=None, help="可选真值标签 0–9")
    parser.add_argument(
        "--tensor-name",
        default="image",
        help="图像张量名称（默认 image；与 activation 文件对齐可用 input）",
    )
    parser.add_argument(
        "--no-mnist-style",
        action="store_true",
        help="仅拉伸到 28×28，不做裁切/居中（旧行为）",
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="使用 pixel/255（1=白），与 MNIST 训练数据相反；仅调试旧文件时用",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pngs = [Path(p) for p in args.png]
    mnist_style = not args.no_mnist_style
    mnist_invert = not args.no_invert

    if args.mnist_digit is not None:
        if pngs:
            print("error: --mnist-digit 与 PNG 路径不能同时使用", file=sys.stderr)
            return 2
        out = args.output or Path(f"mnist_{args.mnist_digit}.inference")
        label = args.label if args.label is not None else args.mnist_digit
        try:
            image = mnist_test_digit_batch(args.mnist_digit)
            tensors: dict[str, np.ndarray] = {args.tensor_name: image}
            tensors["label"] = np.array([float(label)], dtype=np.float32)
            save_infr(out, tensors)
            print(f"Wrote {out}  (MNIST test digit {args.mnist_digit}, {args.tensor_name} {image.shape})")
        except (OSError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if not pngs:
        print("error: 请指定 PNG 或使用 --mnist-digit", file=sys.stderr)
        return 2

    if args.output is not None and len(pngs) != 1:
        print("error: -o/--output 只能用于单个 PNG 输入", file=sys.stderr)
        return 2

    try:
        if len(pngs) == 1:
            out = args.output or pngs[0].with_suffix(".inference")
            convert_one(
                pngs[0],
                out,
                label=args.label,
                tensor_name=args.tensor_name,
                mnist_style=mnist_style,
                mnist_invert=mnist_invert,
            )
        else:
            if args.label is not None:
                print("error: 批量转换时不要用 --label（请逐个转换）", file=sys.stderr)
                return 2
            for png in pngs:
                convert_one(
                    png,
                    png.with_suffix(".inference"),
                    label=None,
                    tensor_name=args.tensor_name,
                    mnist_style=mnist_style,
                    mnist_invert=mnist_invert,
                )
    except (OSError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

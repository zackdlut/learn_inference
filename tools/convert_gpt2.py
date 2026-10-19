#!/usr/bin/env python3
"""把 HuggingFace 的 GPT-2 权重转换成我们的 .infer 格式。

    .venv/bin/python tools/convert_gpt2.py

会下载（约 550 MB，只下一次）：
    model.safetensors   权重
    vocab.json          BPE 词表
    merges.txt          BPE 合并规则

safetensors 格式很简单，比 PyTorch 的 pickle 好解析得多：
    ┌──────────────────────────────────────┐
    │ header_len   uint64 (小端)            │
    │ header       JSON，描述每个张量        │
    │ data         所有张量的原始字节        │
    └──────────────────────────────────────┘
JSON 里每个条目形如：
    "h.0.attn.c_attn.weight": {"dtype":"F32","shape":[768,2304],"data_offsets":[0,7077888]}

这也正是我们阶段 3 设计 .infer 格式时参考的思路：元信息和数据分开放。
"""

import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "gpt"
OUT = ROOT / "data" / "gpt"

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main/"
FILES = ["model.safetensors", "vocab.json", "merges.txt", "config.json"]


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = CACHE / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  {name} 已存在 ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  下载 {name} ...", end=" ", flush=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE + name, timeout=600) as r:
                    data = r.read()
                dst.write_bytes(data)
                print(f"{len(data) / 1e6:.1f} MB")
                break
            except Exception as e:  # noqa: BLE001
                print(f"失败({e})，重试 {attempt + 1}/3 ...", end=" ", flush=True)
        else:
            sys.exit(f"下载 {name} 失败，可手动下载后放到 {CACHE}")


def load_safetensors(path: Path) -> dict:
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    print("header: ", json.dumps(header,indent=4))
    body = raw[8 + header_len :]

    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        assert meta["dtype"] == "F32", f"{name} 是 {meta['dtype']}，本脚本只处理 F32"
        lo, hi = meta["data_offsets"]
        arr = np.frombuffer(body[lo:hi], dtype="<f4").reshape(meta["shape"])
        out[name] = arr
    return out


# --- .infer 格式（和 tools/train_mnist.py 里的完全一样）----------------------

MAGIC, VERSION, DTYPE_F32 = b"INFR", 1, 0


def save_model(path: Path, tensors: dict, quiet=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs, meta, cursor = [], [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        raw = a.tobytes()
        meta.append((name, a.shape, cursor, len(raw)))
        blobs.append(raw)
        cursor += len(raw)

    header = bytearray(MAGIC)
    header += struct.pack("<II", VERSION, len(meta))
    for name, shape, off, nbytes in meta:
        nb = name.encode("utf-8")
        header += struct.pack("<I", len(nb)) + nb
        header += struct.pack("<II", DTYPE_F32, len(shape))
        for d in shape:
            header += struct.pack("<q", d)
        header += struct.pack("<QQ", off, nbytes)
    data_start = len(header) + 8
    header += struct.pack("<Q", data_start)

    with open(path, "wb") as f:
        f.write(header)
        for b in blobs:
            f.write(b)
    if not quiet:
        print(f"  写入 {path.name}: {len(meta)} 个张量, {path.stat().st_size / 1e6:.1f} MB")


# --- 参考实现：用 numpy 跑一遍 GPT-2，用来和 C++ 逐层对拍 ---------------------


def gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def gpt2_forward(w, tokens, n_head=12, collect=None):
    """GPT-2 前向。刻意写得和 C++ 侧一一对应，方便对照阅读。

    HuggingFace 的 GPT-2 用的是 Conv1D 而不是 Linear，权重形状是 (in, out)，
    计算是 x @ W + b —— 注意这和 PyTorch nn.Linear 的 (out, in) 是**转置关系**。
    这是移植 GPT-2 时最容易搞错的地方。
    """
    T = len(tokens)
    C = w["wte.weight"].shape[1]
    hs = C // n_head

    # token embedding + 位置 embedding
    x = w["wte.weight"][tokens] + w["wpe.weight"][:T]
    if collect is not None:
        collect["embed"] = x.copy()

    n_layer = sum(1 for k in w if k.startswith("h.") and k.endswith(".attn.c_attn.weight"))
    for L in range(n_layer):
        p = f"h.{L}."

        # --- 注意力块 ---
        h = layernorm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]  # (T, 3C)
        q, k, v = np.split(qkv, 3, axis=-1)

        # 拆成多头: (T, C) -> (n_head, T, hs)
        q = q.reshape(T, n_head, hs).transpose(1, 0, 2)
        k = k.reshape(T, n_head, hs).transpose(1, 0, 2)
        v = v.reshape(T, n_head, hs).transpose(1, 0, 2)

        att = q @ k.transpose(0, 2, 1) / np.sqrt(hs)          # (n_head, T, T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)       # 因果掩码
        att = np.where(mask, -np.inf, att)
        att = softmax(att)
        y = att @ v                                            # (n_head, T, hs)
        y = y.transpose(1, 0, 2).reshape(T, C)                 # 合并回 (T, C)

        y = y @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        x = x + y                                              # 残差
        if collect is not None and L == 0:
            collect["l0_attn_out"] = x.copy()

        # --- MLP 块 ---
        h = layernorm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        h = h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]   # C -> 4C
        h = gelu(h)
        h = h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]  # 4C -> C
        x = x + h                                              # 残差
        if collect is not None and L == 0:
            collect["l0_out"] = x.copy()

    x = layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
    if collect is not None:
        collect["final_norm"] = x.copy()

    # 输出层和 token embedding 共享权重（weight tying），省下 38M 参数
    logits = x @ w["wte.weight"].T
    return logits


def main():
    print("=== 1. 下载 GPT-2 ===")
    download()

    print("\n=== 2. 解析 safetensors ===")
    w = load_safetensors(CACHE / "model.safetensors")
    cfg = json.loads((CACHE / "config.json").read_text())
    print("cfg: ", cfg)
    n_layer, n_head, n_embd = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
    n_vocab, n_ctx = cfg["vocab_size"], cfg["n_ctx"]
    print(f"  {len(w)} 个张量")
    print(f"  层数 {n_layer}, 注意力头 {n_head}, 隐藏维 {n_embd}, "
          f"词表 {n_vocab}, 上下文 {n_ctx}")
    total = sum(a.size for a in w.values())
    print(f"  参数量 {total / 1e6:.1f} M  (f32 存储 {total * 4 / 1e6:.0f} MB)")

    print("\n=== 3. 转成 .infer 格式 ===")
    out = {}
    # 名字保持和 HuggingFace 一致，C++ 侧按同样的名字取，一一对应不容易出错
    out["wte.weight"] = w["wte.weight"]
    out["wpe.weight"] = w["wpe.weight"]
    out["ln_f.weight"] = w["ln_f.weight"]
    out["ln_f.bias"] = w["ln_f.bias"]
    for L in range(n_layer):
        p = f"h.{L}."
        for suffix in ["ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                       "attn.c_attn.weight", "attn.c_attn.bias",
                       "attn.c_proj.weight", "attn.c_proj.bias",
                       "mlp.c_fc.weight", "mlp.c_fc.bias",
                       "mlp.c_proj.weight", "mlp.c_proj.bias"]:
            out[p + suffix] = w[p + suffix]
    save_model(OUT / "gpt2.inference", out)

    # 超参数单独存一份，C++ 侧读了才知道怎么搭网络
    save_model(OUT / "gpt2_config.inference", {
        "n_layer": np.array([n_layer], dtype=np.float32),
        "n_head": np.array([n_head], dtype=np.float32),
        "n_embd": np.array([n_embd], dtype=np.float32),
        "n_vocab": np.array([n_vocab], dtype=np.float32),
        "n_ctx": np.array([n_ctx], dtype=np.float32),
    })

    print("\n=== 4. 导出分词器 ===")
    vocab = json.loads((CACHE / "vocab.json").read_text(encoding="utf-8"))
    merges = (CACHE / "merges.txt").read_text(encoding="utf-8").split("\n")
    merges = [m for m in merges[1:] if m.strip()]  # 第一行是版本注释

    # 词表存成简单的二进制：每条 [长度 uint32][原始字节]
    # 按 token id 排序，这样 C++ 侧直接按下标取
    id2tok = {v: k for k, v in vocab.items()}
    with open(OUT / "gpt2_vocab.bin", "wb") as f:
        f.write(b"GPTV")
        f.write(struct.pack("<I", len(id2tok)))
        for i in range(len(id2tok)):
            raw = id2tok[i].encode("utf-8")
            f.write(struct.pack("<I", len(raw)) + raw)
    print(f"  写入 gpt2_vocab.bin: {len(id2tok)} 个 token")

    with open(OUT / "gpt2_merges.bin", "wb") as f:
        f.write(b"GPTM")
        f.write(struct.pack("<I", len(merges)))
        for line in merges:
            a, b = line.split(" ")
            for part in (a, b):
                raw = part.encode("utf-8")
                f.write(struct.pack("<I", len(raw)) + raw)
    print(f"  写入 gpt2_merges.bin: {len(merges)} 条合并规则")

    print("\n=== 5. 生成逐层对拍数据 ===")
    # 用一小段 token 跑一遍参考实现，把中间结果存下来
    prompt_tokens = [464, 3139, 286, 4881, 318]  # "The capital of France is"
    collect = {}
    logits = gpt2_forward(w, prompt_tokens, n_head, collect)
    collect["logits"] = logits.astype(np.float32)
    collect["tokens"] = np.array(prompt_tokens, dtype=np.float32)
    save_model(OUT / "gpt2_reference.inference", {k: v.astype(np.float32)
                                              for k, v in collect.items()})

    top = np.argsort(logits[-1])[::-1][:5]
    print("  参考实现对 'The capital of France is' 的 top-5 预测:")
    for t in top:
        text = bytearray()
        s = id2tok[int(t)]
        # GPT-2 用一套 byte-level 编码，Ġ 表示空格
        print(f"    token {int(t):>6}  {s!r:>14}  logit {logits[-1][t]:.3f}")

    print("\n完成。接下来编译并运行: ./build/examples/06_gpt2")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build docs/train_mnist_step_by_step.html: one B=100 batch, matrix schematics + readable slices."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from train_mnist_numpy import MLP, load_data  # noqa: E402

NB, NJ, NP, NT = 4, 6, 6, 4  # shown samples, hidden units, pixel columns, listed terms


def g(v):
    return float(f"{float(v):.6g}")


def arr(a):
    a = np.asarray(a)
    if a.ndim == 2:
        return [[g(v) for v in row] for row in a]
    return [g(v) for v in a]


def top_terms(a, w, k=NT):
    prod = a.astype(np.float64) * w.astype(np.float64)
    order = [int(i) for i in np.argsort(-np.abs(prod))[:k] if prod[i] != 0]
    return {
        "t": [[i, g(a[i]), g(w[i]), g(prod[i])] for i in order],
        "rest": g(prod.sum() - prod[order].sum()),
        "nnz": int((prod != 0).sum()),
        "sum": g(prod.sum()),
    }


def first_terms(a, w, k=NT):
    prod = a.astype(np.float64) * w.astype(np.float64)
    return {
        "t": [[i, g(a[i]), g(w[i]), g(prod[i])] for i in range(k)],
        "rest": g(prod[k:].sum()),
        "sum": g(prod.sum()),
    }


def build_data():
    train_x, train_y, _, _ = load_data()
    m = MLP()
    B, lr = m.batch_size, m.learning_rate
    x, y = train_x[:B], train_y[:B]
    W1, b1, W2, b2 = m.W1.copy(), m.b1.copy(), m.W2.copy(), m.b2.copy()

    z1, h1, z2, p = m.forward(x)
    dW1, db1, dW2, db2 = m.backward(x, y, z1, h1, p)
    onehot = np.zeros_like(p)
    onehot[np.arange(B), y] = 1.0
    delta = (p - onehot) / B
    dh = delta @ W2
    mask = (z1 > 0).astype(np.float32)
    dz1 = dh * mask
    assert np.allclose(dz1.T @ x, dW1, atol=1e-6)
    loss = -np.log(p[np.arange(B), y] + 1e-9)

    W1n, b1n = W1 - lr * dW1, b1 - lr * db1
    W2n, b2n = W2 - lr * dW2, b2 - lr * db2
    m.W1, m.b1, m.W2, m.b2 = W1n, b1n, W2n, b2n
    p_after = m.forward(x)[3]
    loss_after = -np.log(p_after[np.arange(B), y] + 1e-9)

    pix = np.sort(np.argsort(-x[:NB].sum(axis=0))[:NP])
    bs, js, ks = range(NB), range(NJ), range(10)
    sm = []
    for b in bs:
        mx = z2[b].max()
        e = np.exp(z2[b] - mx)
        sm.append({"max": g(mx), "e": arr(e), "s": g(e.sum())})

    return {
        "B": B, "lr": lr, "NB": NB, "NJ": NJ, "pix": pix.tolist(),
        "imgs": [[round(float(v), 2) for v in x[b]] for b in bs],
        "y": y[:NB].tolist(),
        "x": arr(x[:NB][:, pix]),
        "W1": arr(W1[:NJ][:, pix]), "b1": arr(b1[:NJ]),
        "z1": arr(z1[:NB, :NJ]), "h1": arr(h1[:NB, :NJ]),
        "z1T": [[top_terms(x[b], W1[j]) for j in js] for b in bs],
        "W2": arr(W2[:, :NJ]), "b2": arr(b2),
        "z2": arr(z2[:NB]), "p": arr(p[:NB]), "sm": sm,
        "z2T": [[top_terms(h1[b], W2[k]) for k in ks] for b in bs],
        "loss": arr(loss[:NB]), "lossSum": g(loss.sum()), "lossMean": g(loss.mean()),
        "onehot": arr(onehot[:NB]), "delta": arr(delta[:NB]),
        "dW2": arr(dW2[:, :NJ]),
        "dW2T": [[first_terms(delta[:, k], h1[:, j]) for j in js] for k in ks],
        "db2": arr(db2),
        "db2T": [{"t": [g(delta[b, k]) for b in bs], "rest": g(delta[NB:, k].sum())} for k in ks],
        "dh": arr(dh[:NB, :NJ]),
        "dhT": [[first_terms(delta[b], W2[:, j], k=10) for j in js] for b in bs],
        "mask": arr(mask[:NB, :NJ]), "dz1": arr(dz1[:NB, :NJ]),
        "dW1": arr(dW1[:NJ][:, pix]),
        "dW1T": [[first_terms(dz1[:, j], x[:, i]) for i in pix] for j in js],
        "db1": arr(db1[:NJ]),
        "db1T": [{"t": [g(dz1[b, j]) for b in bs], "rest": g(dz1[NB:, j].sum())} for j in js],
        "W1n": arr(W1n[:NJ][:, pix]), "b1n": arr(b1n[:NJ]),
        "W2n": arr(W2n[:, :NJ]), "b2n": arr(b2n),
        "pAfter": arr(p_after[:NB]), "lossAfter": arr(loss_after[:NB]),
        "lossAfterMean": g(loss_after.mean()),
        "acc": g((p.argmax(1) == y).mean()), "accAfter": g((p_after.argmax(1) == y).mean()),
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MNIST MLP 一个 batch 的推理与训练（B=100）</title>
<style>
  :root {
    --bg: #0f1117; --panel: #171a22; --stroke: #2a3140; --text: #e8eaef; --muted: #8b93a7;
    --accent: #5b9fd4; --hi: #e8c54b; --pos: #7bc96f; --neg: #d47b9f;
    --mono: "JetBrains Mono", "IBM Plex Mono", Consolas, monospace;
    --sans: "IBM Plex Sans", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 15px; }
  .app { display: grid; grid-template-columns: 230px minmax(0, 1fr); min-height: 100vh; }
  nav { border-right: 1px solid var(--stroke); padding: 0.75rem 0.6rem; background: #12151c; position: sticky; top: 0; height: 100vh; overflow-y: auto; }
  nav h1 { font-size: 0.95rem; margin: 0 0 0.75rem; line-height: 1.4; }
  nav h2 { font-size: 0.72rem; color: var(--muted); margin: 0.9rem 0 0.3rem; letter-spacing: 0.04em; }
  nav button { display: block; width: 100%; text-align: left; background: none; border: 1px solid transparent; color: var(--text);
    padding: 0.3rem 0.45rem; border-radius: 5px; font: inherit; font-size: 0.85rem; cursor: pointer; }
  nav button:hover { border-color: var(--stroke); }
  nav button.on { border-color: var(--accent); color: var(--accent); }
  main { padding: 0.9rem 1.25rem 2rem; min-width: 0; }
  .bar { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 0.75rem; }
  .bar button, .bar select { background: var(--panel); border: 1px solid var(--stroke); color: var(--text); padding: 0.4rem 0.8rem;
    border-radius: 6px; font: inherit; font-size: 0.9rem; cursor: pointer; }
  .bar button.primary { background: var(--accent); border-color: var(--accent); color: #0f1117; font-weight: 600; }
  .bar .info { color: var(--muted); font-size: 0.85rem; margin-left: 0.5rem; }
  .phase { font-size: 0.75rem; color: var(--accent); letter-spacing: 0.05em; }
  h2#stitle { margin: 0.15rem 0 0.5rem; font-size: 1.3rem; }
  pre#code { background: var(--panel); border: 1px solid var(--stroke); border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0 0 0.5rem;
    font-family: var(--mono); font-size: 0.9rem; color: #a9c7e8; overflow-x: auto; }
  #desc { color: #c5cad6; line-height: 1.7; margin: 0 0 0.75rem; max-width: 70rem; }
  .schem { background: var(--panel); border: 1px solid var(--stroke); border-radius: 10px; padding: 0.5rem; margin-bottom: 0.9rem; text-align: center; }
  .schem svg { max-width: 100%; max-height: 250px; }
  .schem .cap { font-size: 0.75rem; color: var(--muted); }
  svg .mat { fill: #1d2330; stroke: #4a5568; stroke-width: 1.5; }
  svg .mat.act { stroke: var(--accent); }
  svg .hlr { fill: rgba(91, 159, 212, 0.55); }
  svg .hlc { fill: var(--hi); }
  svg .mn { fill: var(--text); font-size: 15px; font-weight: 600; text-anchor: middle; font-family: var(--mono); }
  svg .ms { fill: var(--accent); font-size: 13px; text-anchor: middle; font-family: var(--mono); }
  svg .md { fill: var(--muted); font-size: 11px; text-anchor: middle; }
  svg .op { fill: var(--muted); font-size: 18px; text-anchor: middle; font-family: var(--mono); }
  .work { display: grid; grid-template-columns: minmax(0, 1fr) minmax(340px, 470px); gap: 1rem; align-items: start; }
  @media (max-width: 1250px) { .work { grid-template-columns: 1fr; } }
  #tables { display: flex; flex-wrap: wrap; gap: 0.9rem 1.1rem; align-items: flex-start; }
  .tbl .tt { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.25rem; }
  .tbl .note { font-size: 0.75rem; color: var(--muted); margin-top: 0.15rem; }
  table { border-collapse: collapse; font-family: var(--mono); font-size: 14px; }
  th { color: var(--muted); font-weight: 400; font-size: 12px; padding: 3px 6px; }
  td { border: 1px solid #262d3b; padding: 4px 8px; text-align: right; min-width: 64px; background: #141821; }
  td.more { color: var(--muted); text-align: center; border-style: dashed; min-width: 30px; background: none; }
  td.neg { color: var(--neg); }
  td.zero { color: #59617a; }
  td.target { box-shadow: inset 0 0 0 2px var(--pos); }
  td.in { background: #1c3048; }
  td.out { background: #4a3f12; outline: 2px solid var(--hi); outline-offset: -2px; color: #fff; }
  td.click { cursor: pointer; text-decoration: underline dotted #4a5568; }
  .imgs { display: flex; gap: 0.6rem; }
  .imgs figure { margin: 0; text-align: center; font-size: 0.8rem; color: var(--muted); }
  .imgs canvas { width: 140px; height: 140px; image-rendering: pixelated; border: 2px solid var(--stroke); border-radius: 6px; display: block; }
  .imgs figure.sel canvas { border-color: var(--hi); }
  .side { position: sticky; top: 0.75rem; }
  .formula { background: var(--panel); border: 1px solid var(--stroke); border-radius: 10px; padding: 0.8rem 1rem; font-family: var(--mono);
    font-size: 14.5px; line-height: 1.75; white-space: pre-wrap; word-break: break-word; }
  .formula .v { color: #a9c7e8; }
  .formula .res { color: var(--hi); font-size: 1.1em; }
  .legend { font-size: 0.78rem; color: var(--muted); margin-top: 0.6rem; line-height: 1.8; }
  .legend i { display: inline-block; width: 14px; height: 12px; vertical-align: middle; margin-right: 4px; border-radius: 2px; }
</style>
</head>
<body>
<div class="app">
  <nav id="nav"><h1>一个 batch 的推理与训练<br><span style="color:var(--muted);font-weight:400">B = 100 · lr = 0.1</span></h1></nav>
  <main>
    <div class="bar">
      <button type="button" id="prevStep">« 上一步</button>
      <button type="button" id="prevCell">‹ 上一个元素</button>
      <button type="button" class="primary" id="play">播放</button>
      <button type="button" id="nextCell">下一个元素 ›</button>
      <button type="button" id="nextStep">下一步 »</button>
      <select id="speed">
        <option value="3000">慢</option>
        <option value="1800" selected>中</option>
        <option value="900">快</option>
      </select>
      <span class="info" id="cellInfo"></span>
    </div>
    <div class="phase" id="phase"></div>
    <h2 id="stitle"></h2>
    <pre id="code"></pre>
    <p id="desc"></p>
    <div class="schem"><div id="schem"></div>
      <div class="cap">示意图：方块按真实形状绘制（边长取对数，否则 784 会把 10 挤没）；蓝色条 = 参与本次计算的行/列，黄点 = 正在计算的输出元素</div>
    </div>
    <div class="work">
      <div id="tables"></div>
      <div class="side">
        <div class="formula" id="formula"></div>
        <div class="legend">
          <i style="background:#1c3048"></i>参与计算的输入
          <i style="background:#4a3f12;outline:2px solid var(--hi)"></i>当前输出元素
          <i style="box-shadow:inset 0 0 0 2px var(--pos)"></i>真实类别<br>
          表格只放大真实矩阵的一角，⋮ ⋯ 表示省略的行列；公式里的求和用的是完整矩阵。键盘 ← → 切换元素。
        </div>
      </div>
    </div>
  </main>
</div>
<script>
const D = __DATA__;

const range = (n) => Array.from({ length: n }, (_, i) => i);
function fmt(v) {
  if (typeof v !== 'number') return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  return String(Number(v.toPrecision(4)));
}
const pv = (v) => (v < 0 ? '(' + fmt(v) + ')' : fmt(v));
const V = (v) => `<span class="v">${fmt(v)}</span>`;
const RES = (v) => `<b class="res">${fmt(v)}</b>`;
const vec = (a) => '[' + a.map(fmt).join(', ') + ']';
const NB = D.NB, NJ = D.NJ, NP = D.pix.length, B = D.B;
const bL = range(NB).map((b) => 'b=' + b);
const jL = range(NJ).map((j) => 'j=' + j);
const kL = range(10).map((k) => 'k=' + k);
const iL = D.pix.map((i) => 'i=' + i);
const col = (a) => a.map((v) => [v]);
const T = (id, title, rows, cols, data, o) => Object.assign({ id, title, rows, cols, data }, o || {});
const M = (n, r, c, d) => ({ n, r, c, d });
const OP = (op) => ({ op });
const R100 = '⋮ 共 100 行';
const isTarget = (r, c) => (c === D.y[r] ? 'target' : '');
const cells2 = (tid, R, C) => {
  const a = [];
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) a.push({ at: [tid, r, c] });
  return a;
};
const argmax = (a) => a.indexOf(Math.max(...a));
const restB = `b=${NB}..${B - 1} 共 ${B - NB} 项之和`;

const tb = {
  img: { id: 'img', kind: 'img' },
  x: T('x', 'X：样本 0..3 × 6 个像素列', bL, iL, D.x, { moreR: R100, moreC: '⋯ 共 784 列' }),
  y: T('y', 'y', bL, ['标签'], col(D.y), { moreR: '⋮' }),
  W1: T('W1', 'W₁（存储 128×784，第 j 行 = 神经元 j 的权重）', jL, iL, D.W1, { moreR: '⋮ 共 128 行', moreC: '⋯ 共 784 列' }),
  b1: T('b1', 'b₁', ['b₁'], jL, [D.b1], { moreC: '⋯ 共 128 个' }),
  z1: T('z1', 'Z₁', bL, jL, D.z1, { moreR: R100, moreC: '⋯ 共 128 列', cls: (r, c, v) => (v < 0 ? 'neg' : '') }),
  h1: T('h1', 'H₁ = ReLU(Z₁)', bL, jL, D.h1, { moreR: R100, moreC: '⋯ 共 128 列', cls: (r, c, v) => (v === 0 ? 'zero' : '') }),
  W2: T('W2', 'W₂（存储 10×128）', kL, jL, D.W2, { moreC: '⋯ 共 128 列' }),
  b2: T('b2', 'b₂', ['b₂'], kL, [D.b2]),
  z2: T('z2', 'Z₂（logits）', bL, kL, D.z2, { moreR: R100 }),
  p: T('p', 'P = softmax(Z₂)', bL, kL, D.p, { moreR: R100, cls: isTarget }),
  loss: T('loss', '−log P[b, y_b]', bL, ['loss'], col(D.loss), { moreR: R100 }),
  onehot: T('onehot', 'Y_onehot', bL, kL, D.onehot, { moreR: R100, cls: isTarget }),
  delta: T('delta', 'δ = (P − Y) / 100', bL, kL, D.delta, { moreR: R100, cls: isTarget }),
  dW2: T('dW2', 'dW₂', kL, jL, D.dW2, { moreC: '⋯ 共 128 列' }),
  db2: T('db2', 'db₂', ['db₂'], kL, [D.db2]),
  dh: T('dh', 'δ_h = δ @ W₂', bL, jL, D.dh, { moreR: R100, moreC: '⋯ 共 128 列' }),
  mask: T('mask', '掩码 (Z₁ > 0)', bL, jL, D.mask, { moreR: R100, moreC: '⋯ 共 128 列', cls: (r, c, v) => (v === 0 ? 'zero' : '') }),
  dz1: T('dz1', 'δ_z₁', bL, jL, D.dz1, { moreR: R100, moreC: '⋯ 共 128 列', cls: (r, c, v) => (v === 0 ? 'zero' : '') }),
  dW1: T('dW1', 'dW₁', jL, iL, D.dW1, { moreR: '⋮ 共 128 行', moreC: '⋯ 共 784 列' }),
  db1: T('db1', 'db₁', ['db₁'], jL, [D.db1], { moreC: '⋯ 共 128 个' }),
  W2n: T('W2n', "W₂'（更新后）", kL, jL, D.W2n, { moreC: '⋯ 共 128 列' }),
  b2n: T('b2n', "b₂'（更新后）", ['b₂\''], kL, [D.b2n]),
  W1n: T('W1n', "W₁'（更新后）", jL, iL, D.W1n, { moreR: '⋮ 共 128 行', moreC: '⋯ 共 784 列' }),
  b1n: T('b1n', "b₁'（更新后）", ['b₁\''], jL, [D.b1n], { moreC: '⋯ 共 128 个' }),
  pAfter: T('pAfter', "P'（新参数对同一 batch 再前向）", bL, kL, D.pAfter, { moreR: R100, cls: isTarget }),
  lc: T('lc', '前 4 个样本的 loss', bL, ['更新前', '更新后'], range(NB).map((b) => [D.loss[b], D.lossAfter[b]])),
};

function sgdText(name, gname, r, c, old, g, nw) {
  const idx = c == null ? `[${r}]` : `[${r},${c}]`;
  return `<b>${name}${idx}</b> ← ${name}${idx} − lr × ${gname}${idx}\n` +
    `  = ${fmt(old)} − 0.1 × ${pv(g)}\n  = ${RES(nw)}\n\n` +
    `变化量 = ${fmt(nw - old)}\n梯度 > 0 → 参数变小；梯度 < 0 → 参数变大。\n沿负梯度方向走一小步，让本 batch 的 loss 下降。`;
}

const STEPS = [
  {
    phase: '前向（推理）', title: '取一个 mini-batch',
    code: 'x_batch = x[i : i + 100]    # (100, 784)\ny_batch = y[i : i + 100]    # (100,)',
    desc: '训练循环每次从 60000 张图里取 <b>100</b> 张（batch_size = 100）。每张 28×28 的图展平成 784 维行向量，像素 ÷255 归一化到 [0,1]，100 张叠起来就是矩阵 <b>X (100×784)</b>。下面放大<b>前 4 个样本</b>和 <b>6 个像素列</b>（图上黄框）。',
    schem: [M('X', 100, 784, '样本 × 像素'), OP('  '), M('y', 100, 1, '标签')],
    tables: [tb.img, tb.x, tb.y], out: ['x'], cells: cells2('x', NB, NP),
    focus: (b, c) => {
      const i = D.pix[c];
      return {
        hl: { img: [[b, c, 'out']], x: [[b, c, 'out']], y: [[b, 0, 'in']] },
        sch: { 0: { cell: [b, i] }, 2: { row: b } },
        f: `<b>x[${b}, ${i}]</b> = 第 ${b} 张图 第 ${Math.floor(i / 28)} 行 第 ${i % 28} 列的像素 ÷ 255\n  = ${RES(D.x[b][c])}\n\n` +
          `像素下标 i = 行 × 28 + 列 = ${Math.floor(i / 28)} × 28 + ${i % 28} = ${i}\n这张图的标签 y[${b}] = <b>${D.y[b]}</b>`,
      };
    },
  },
  {
    phase: '前向（推理）', title: '第一层线性：Z₁ = X · W₁ᵀ + b₁',
    code: 'z1 = x @ self.W1.T + self.b1    # (100,784) @ (784,128) + (128,) → (100,128)',
    desc: 'W₁ 按 nn.Linear 的约定存成 <b>(128, 784)</b>，第 j 行是隐藏神经元 j 的 784 个权重。<b>Z₁[b, j]</b> = 样本 b 的 784 个像素与 W₁ 第 j 行做点积，再加偏置 b₁[j]。b₁ 自动广播到全部 100 行。',
    schem: [M('X', 100, 784, '样本 × 像素'), OP('@'), M('W₁ᵀ', 784, 128, '像素 × 神经元'), OP('+'), M('b₁', 1, 128, '广播到每行'), OP('='), M('Z₁', 100, 128, '样本 × 神经元')],
    tables: [tb.x, tb.W1, tb.b1, tb.z1], out: ['z1'], cells: cells2('z1', NB, NJ),
    focus: (b, j) => {
      const t = D.z1T[b][j];
      return {
        hl: { x: [[b, -1, 'in']], W1: [[j, -1, 'in']], b1: [[0, j, 'in']], z1: [[b, j, 'out']] },
        sch: { 0: { row: b }, 2: { col: j }, 4: { col: j }, 6: { cell: [b, j] } },
        f: `<b>z₁[${b},${j}]</b> = Σ<sub>i=0..783</sub> x[${b},i] · W₁[${j},i] + b₁[${j}]\n\n` +
          `784 项里只有 ${t.nnz} 项非零（大部分像素是 0）。\n贡献最大的 ${t.t.length} 项：\n` +
          t.t.map(([i, a, w, p]) => `  x[${b},${i}]·W₁[${j},${i}] = ${pv(a)} × ${pv(w)} = ${V(p)}`).join('\n') +
          `\n  其余 ${t.nnz - t.t.length} 个非零项之和 = ${V(t.rest)}\n` +
          `点积合计 = ${V(t.sum)}\n+ b₁[${j}] = ${V(D.b1[j])}\n= ${RES(D.z1[b][j])}`,
      };
    },
  },
  {
    phase: '前向（推理）', title: 'ReLU：H₁ = max(0, Z₁)',
    code: 'h1 = self.relu(z1)    # np.maximum(0, z1)，逐元素，形状不变 (100,128)',
    desc: '逐元素运算：负数变 0（这个神经元对该样本"不激活"），正数原样保留。红色是负数，灰色是被置 0 的元素。',
    schem: [M('Z₁', 100, 128), OP('ReLU →'), M('H₁', 100, 128)],
    tables: [tb.z1, tb.h1], out: ['h1'], cells: cells2('h1', NB, NJ),
    focus: (b, j) => {
      const z = D.z1[b][j];
      return {
        hl: { z1: [[b, j, 'in']], h1: [[b, j, 'out']] },
        sch: { 0: { cell: [b, j] }, 2: { cell: [b, j] } },
        f: `<b>h₁[${b},${j}]</b> = max(0, z₁[${b},${j}])\n  = max(0, ${fmt(z)})\n  = ${RES(D.h1[b][j])}\n\n` +
          (z > 0 ? 'z₁ > 0：原样通过。' : 'z₁ ≤ 0：被截断为 0，反向传播时这里的梯度也会是 0。'),
      };
    },
  },
  {
    phase: '前向（推理）', title: '第二层线性：Z₂ = H₁ · W₂ᵀ + b₂',
    code: 'z2 = h1 @ self.W2.T + self.b2    # (100,128) @ (128,10) + (10,) → (100,10)',
    desc: '和第一层同样的套路：<b>Z₂[b, k]</b> = 样本 b 的 128 个隐藏激活与 W₂ 第 k 行做点积，再加 b₂[k]。每个样本得到 10 个 logit，对应数字 0~9。',
    schem: [M('H₁', 100, 128, '样本 × 神经元'), OP('@'), M('W₂ᵀ', 128, 10), OP('+'), M('b₂', 1, 10), OP('='), M('Z₂', 100, 10, '样本 × 类别')],
    tables: [tb.h1, tb.W2, tb.b2, tb.z2], out: ['z2'], cells: cells2('z2', NB, 10),
    focus: (b, k) => {
      const t = D.z2T[b][k];
      return {
        hl: { h1: [[b, -1, 'in']], W2: [[k, -1, 'in']], b2: [[0, k, 'in']], z2: [[b, k, 'out']] },
        sch: { 0: { row: b }, 2: { col: k }, 4: { col: k }, 6: { cell: [b, k] } },
        f: `<b>z₂[${b},${k}]</b> = Σ<sub>j=0..127</sub> h₁[${b},j] · W₂[${k},j] + b₂[${k}]\n\n` +
          `128 项里 h₁ 非零的有 ${t.nnz} 项。贡献最大的 ${t.t.length} 项：\n` +
          t.t.map(([j, a, w, p]) => `  h₁[${b},${j}]·W₂[${k},${j}] = ${pv(a)} × ${pv(w)} = ${V(p)}`).join('\n') +
          `\n  其余 ${t.nnz - t.t.length} 项之和 = ${V(t.rest)}\n` +
          `点积合计 = ${V(t.sum)}\n+ b₂[${k}] = ${V(D.b2[k])}\n= ${RES(D.z2[b][k])}`,
      };
    },
  },
  {
    phase: '前向（推理）', title: 'Softmax：P = softmax(Z₂)',
    code: 'x = z2 - np.max(z2, axis=1, keepdims=True)    # 数值稳定\ne = np.exp(x)\np = e / np.sum(e, axis=1, keepdims=True)        # (100,10)，每行和为 1',
    desc: '按<b>行</b>把 10 个 logit 变成概率。先减去该行最大值防止 exp 溢出（结果不变）。<b>推理（predict）到这里就结束了</b>：取每行最大概率的类别。绿框是真实类别。',
    schem: [M('Z₂', 100, 10), OP('按行 softmax →'), M('P', 100, 10, '每行和 = 1')],
    tables: [tb.z2, tb.p], out: ['p'], cells: cells2('p', NB, 10),
    focus: (b, k) => {
      const s = D.sm[b];
      return {
        hl: { z2: [[b, -1, 'in']], p: [[b, -1, 'in'], [b, k, 'out']] },
        sch: { 0: { row: b }, 2: { cell: [b, k] } },
        f: `样本 ${b} 的 10 个 logit：\n z₂[${b}] = ${vec(D.z2[b])}\n max = ${V(s.max)}\n` +
          ` z − max = ${vec(D.z2[b].map((v) => v - s.max))}\n exp(·) = ${vec(s.e)}\n Σ exp = ${V(s.s)}\n\n` +
          `<b>p[${b},${k}]</b> = exp(z₂[${b},${k}] − max) / Σ\n  = ${fmt(s.e[k])} / ${fmt(s.s)} = ${RES(D.p[b][k])}\n\n` +
          `预测 = argmax p[${b}] = <b>${argmax(D.p[b])}</b>，真实标签 = <b>${D.y[b]}</b>\n（参数还是随机初始化的，基本在瞎猜）`,
      };
    },
  },
  {
    phase: '损失', title: '交叉熵损失',
    code: 'loss += -np.log(p[np.arange(batch_size), y_batch] + 1e-9).sum()',
    desc: '每个样本只看<b>真实类别</b>的概率：概率越接近 1，−log 越接近 0；概率越小惩罚越大。100 个样本的 loss 求和后累加，epoch 结束时除以 60000 打印平均值。',
    schem: [M('P', 100, 10), OP('取真类列 →'), M('−log p', 100, 1), OP('Σ →'), M('L', 1, 1, '标量')],
    tables: [tb.p, tb.loss], out: ['loss'], cells: cells2('loss', NB, 1),
    focus: (b) => ({
      hl: { p: [[b, D.y[b], 'in']], loss: [[b, 0, 'out']] },
      sch: { 0: { cell: [b, D.y[b]] }, 2: { row: b } },
      f: `样本 ${b} 的真实标签 y[${b}] = ${D.y[b]}\n p[${b}, ${D.y[b]}] = ${V(D.p[b][D.y[b]])}\n\n` +
        `<b>loss[${b}]</b> = −log(${fmt(D.p[b][D.y[b]])} + 1e−9) = ${RES(D.loss[b])}\n\n` +
        `整个 batch：Σ loss = ${V(D.lossSum)}，平均 = ${V(D.lossMean)}\n` +
        `（10 类均匀瞎猜时 −log(0.1) ≈ 2.303）\n本 batch 准确率 = ${fmt(D.acc * 100)}%`,
    }),
  },
  {
    phase: '反向传播', title: '输出层误差：δ = (P − Y_onehot) / B',
    code: 'y_onehot[np.arange(batch_size), y_batch] = 1.0\ndelta = (p - y_onehot) / batch_size    # (100,10)',
    desc: 'softmax + 交叉熵对 z₂ 的梯度化简后就是 <b>p − y</b>。真实类别位置为负（要把它推高），其他位置为正（要压低）。除以 B = 100 相当于对 batch 取平均。',
    schem: [M('P', 100, 10), OP('−'), M('Y_onehot', 100, 10), OP('÷ 100 ='), M('δ', 100, 10, '∂L/∂Z₂')],
    tables: [tb.p, tb.onehot, tb.delta], out: ['delta'], cells: cells2('delta', NB, 10),
    focus: (b, k) => ({
      hl: { p: [[b, k, 'in']], onehot: [[b, k, 'in']], delta: [[b, k, 'out']] },
      sch: { 0: { cell: [b, k] }, 2: { cell: [b, k] }, 4: { cell: [b, k] } },
      f: `<b>δ[${b},${k}]</b> = (p[${b},${k}] − y_onehot[${b},${k}]) / 100\n  = (${fmt(D.p[b][k])} − ${D.onehot[b][k]}) / 100\n  = ${RES(D.delta[b][k])}\n\n` +
        (k === D.y[b] ? `k = ${k} 是真实类别 → δ < 0，梯度下降会把 z₂[${b},${k}] 推高。` : `k = ${k} 不是真实类别 → δ > 0，梯度下降会把 z₂[${b},${k}] 压低。`),
    }),
  },
  {
    phase: '反向传播', title: '第二层权重梯度：dW₂ = δᵀ · H₁',
    code: 'dW2 = delta.T @ h1    # (10,100) @ (100,128) → (10,128)，与 W2 同形状',
    desc: '<b>dW₂[k, j]</b> 把 100 个样本的 δ[b,k]·h₁[b,j] 加起来：输出 k 的误差越大、隐藏单元 j 越活跃，W₂[k,j] 的梯度越大。注意这里是对<b>样本维度</b>求和，δ 和 H₁ 都取一整列。',
    schem: [M('δᵀ', 10, 100, '类别 × 样本'), OP('@'), M('H₁', 100, 128, '样本 × 神经元'), OP('='), M('dW₂', 10, 128)],
    tables: [tb.delta, tb.h1, tb.dW2], out: ['dW2'], cells: cells2('dW2', 10, NJ),
    focus: (k, j) => {
      const t = D.dW2T[k][j];
      return {
        hl: { delta: [[-1, k, 'in']], h1: [[-1, j, 'in']], dW2: [[k, j, 'out']] },
        sch: { 0: { row: k }, 2: { col: j }, 4: { cell: [k, j] } },
        f: `<b>dW₂[${k},${j}]</b> = Σ<sub>b=0..99</sub> δ[b,${k}] · h₁[b,${j}]\n\n` +
          t.t.map(([b, a, w, p]) => `  b=${b}: ${pv(a)} × ${pv(w)} = ${V(p)}`).join('\n') +
          `\n  ${restB} = ${V(t.rest)}\n= ${RES(D.dW2[k][j])}`,
      };
    },
  },
  {
    phase: '反向传播', title: '第二层偏置梯度：db₂ = Σ_b δ',
    code: 'db2 = delta.sum(axis=0)    # (10,)',
    desc: '偏置对每个样本都加了一次，所以它的梯度就是 δ 按列（对样本）求和。',
    schem: [M('δ', 100, 10), OP('按列求和 →'), M('db₂', 1, 10)],
    tables: [tb.delta, tb.db2], out: ['db2'], cells: cells2('db2', 1, 10),
    focus: (r, k) => {
      const t = D.db2T[k];
      return {
        hl: { delta: [[-1, k, 'in']], db2: [[0, k, 'out']] },
        sch: { 0: { col: k }, 2: { cell: [0, k] } },
        f: `<b>db₂[${k}]</b> = Σ<sub>b=0..99</sub> δ[b,${k}]\n  = ${t.t.map(pv).join(' + ')}\n    + (${restB} = ${fmt(t.rest)})\n  = ${RES(D.db2[k])}`,
      };
    },
  },
  {
    phase: '反向传播', title: '误差传回隐藏层：δ_h = δ · W₂',
    code: 'delta = (delta @ self.W2) * (z1 > 0)    # 先看 delta @ W2：(100,10) @ (10,128) → (100,128)',
    desc: '把输出层误差沿着 W₂ <b>反方向</b>传回去：隐藏单元 j 收到的误差 = Σ_k δ[b,k]·W₂[k,j]。前向用的是 W₂ᵀ，反向用 W₂ 本身。这里只有 10 项，全部列出。',
    schem: [M('δ', 100, 10), OP('@'), M('W₂', 10, 128), OP('='), M('δ_h', 100, 128, '∂L/∂H₁')],
    tables: [tb.delta, tb.W2, tb.dh], out: ['dh'], cells: cells2('dh', NB, NJ),
    focus: (b, j) => {
      const t = D.dhT[b][j];
      return {
        hl: { delta: [[b, -1, 'in']], W2: [[-1, j, 'in']], dh: [[b, j, 'out']] },
        sch: { 0: { row: b }, 2: { col: j }, 4: { cell: [b, j] } },
        f: `<b>δ_h[${b},${j}]</b> = Σ<sub>k=0..9</sub> δ[${b},k] · W₂[k,${j}]\n\n` +
          t.t.map(([k, a, w, p]) => `  k=${k}: ${pv(a)} × ${pv(w)} = ${V(p)}`).join('\n') +
          `\n= ${RES(D.dh[b][j])}`,
      };
    },
  },
  {
    phase: '反向传播', title: '经过 ReLU：δ_z₁ = δ_h ⊙ (Z₁ > 0)',
    code: 'delta = (delta @ self.W2) * (z1 > 0)    # 逐元素乘 ReLU 导数',
    desc: 'ReLU 的导数：z₁ > 0 处为 1，否则为 0。前向时被 ReLU 截断的神经元，反向时梯度也被截断（灰色）。',
    schem: [M('δ_h', 100, 128), OP('⊙'), M('Z₁>0', 100, 128, '0/1 掩码'), OP('='), M('δ_z₁', 100, 128, '∂L/∂Z₁')],
    tables: [tb.dh, tb.mask, tb.dz1], out: ['dz1'], cells: cells2('dz1', NB, NJ),
    focus: (b, j) => ({
      hl: { dh: [[b, j, 'in']], mask: [[b, j, 'in']], dz1: [[b, j, 'out']] },
      sch: { 0: { cell: [b, j] }, 2: { cell: [b, j] }, 4: { cell: [b, j] } },
      f: `<b>δ_z₁[${b},${j}]</b> = δ_h[${b},${j}] × 1[z₁[${b},${j}] > 0]\n  = ${pv(D.dh[b][j])} × ${D.mask[b][j]}\n  = ${RES(D.dz1[b][j])}\n\n` +
        `前向时 z₁[${b},${j}] = ${fmt(D.z1[b][j])}` + (D.mask[b][j] ? ' > 0，梯度通过。' : ' ≤ 0，梯度被截断。'),
    }),
  },
  {
    phase: '反向传播', title: '第一层权重梯度：dW₁ = δ_z₁ᵀ · X',
    code: 'dW1 = delta.T @ x_batch    # (128,100) @ (100,784) → (128,784)，与 W1 同形状',
    desc: '<b>dW₁[j, i]</b> = 100 个样本的 δ_z₁[b,j]·x[b,i] 之和。像素 i 是 0 的样本对这个梯度没有贡献——所以图片边缘那些永远为 0 的像素，对应的权重永远不会被更新。',
    schem: [M('δ_z₁ᵀ', 128, 100, '神经元 × 样本'), OP('@'), M('X', 100, 784, '样本 × 像素'), OP('='), M('dW₁', 128, 784)],
    tables: [tb.dz1, tb.x, tb.dW1], out: ['dW1'], cells: cells2('dW1', NJ, NP),
    focus: (j, c) => {
      const t = D.dW1T[j][c], i = D.pix[c];
      return {
        hl: { dz1: [[-1, j, 'in']], x: [[-1, c, 'in']], dW1: [[j, c, 'out']] },
        sch: { 0: { row: j }, 2: { col: i }, 4: { cell: [j, i] } },
        f: `<b>dW₁[${j},${i}]</b> = Σ<sub>b=0..99</sub> δ_z₁[b,${j}] · x[b,${i}]\n\n` +
          t.t.map(([b, a, w, p]) => `  b=${b}: ${pv(a)} × ${pv(w)} = ${V(p)}`).join('\n') +
          `\n  ${restB} = ${V(t.rest)}\n= ${RES(D.dW1[j][c])}`,
      };
    },
  },
  {
    phase: '反向传播', title: '第一层偏置梯度：db₁ = Σ_b δ_z₁',
    code: 'db1 = delta.sum(axis=0)    # (128,)',
    desc: '和 db₂ 一样：把 δ_z₁ 按列（对样本）求和。',
    schem: [M('δ_z₁', 100, 128), OP('按列求和 →'), M('db₁', 1, 128)],
    tables: [tb.dz1, tb.db1], out: ['db1'], cells: cells2('db1', 1, NJ),
    focus: (r, j) => {
      const t = D.db1T[j];
      return {
        hl: { dz1: [[-1, j, 'in']], db1: [[0, j, 'out']] },
        sch: { 0: { col: j }, 2: { cell: [0, j] } },
        f: `<b>db₁[${j}]</b> = Σ<sub>b=0..99</sub> δ_z₁[b,${j}]\n  = ${t.t.map(pv).join(' + ')}\n    + (${restB} = ${fmt(t.rest)})\n  = ${RES(D.db1[j])}`,
      };
    },
  },
  {
    phase: 'SGD 更新', title: '更新第二层：W₂ ← W₂ − lr·dW₂，b₂ ← b₂ − lr·db₂',
    code: 'self.W2 -= self.learning_rate * dW2\nself.b2 -= self.learning_rate * db2',
    desc: '每个元素各自独立更新：新值 = 旧值 − 0.1 × 梯度。W₂ 共 10×128 = 1280 个元素，b₂ 共 10 个。',
    schem: [M('W₂', 10, 128), OP('− 0.1 ×'), M('dW₂', 10, 128), OP('='), M("W₂'", 10, 128)],
    tables: [tb.W2, tb.dW2, tb.W2n, tb.b2, tb.db2, tb.b2n], out: ['W2n', 'b2n'],
    cells: cells2('W2n', 10, NJ).concat(cells2('b2n', 1, 10)),
    focus: (r, c, cell) => {
      if (cell.at[0] === 'b2n') {
        return {
          hl: { b2: [[0, c, 'in']], db2: [[0, c, 'in']], b2n: [[0, c, 'out']] }, sch: {},
          f: sgdText('b₂', 'db₂', c, null, D.b2[c], D.db2[c], D.b2n[c]),
        };
      }
      return {
        hl: { W2: [[r, c, 'in']], dW2: [[r, c, 'in']], W2n: [[r, c, 'out']] },
        sch: { 0: { cell: [r, c] }, 2: { cell: [r, c] }, 4: { cell: [r, c] } },
        f: sgdText('W₂', 'dW₂', r, c, D.W2[r][c], D.dW2[r][c], D.W2n[r][c]),
      };
    },
  },
  {
    phase: 'SGD 更新', title: '更新第一层：W₁ ← W₁ − lr·dW₁，b₁ ← b₁ − lr·db₁',
    code: 'self.W1 -= self.learning_rate * dW1\nself.b1 -= self.learning_rate * db1',
    desc: 'W₁ 共 128×784 = 100352 个元素，每个都按同一公式更新。这一步做完，一次训练迭代（一个 batch）就结束了。',
    schem: [M('W₁', 128, 784), OP('− 0.1 ×'), M('dW₁', 128, 784), OP('='), M("W₁'", 128, 784)],
    tables: [tb.W1, tb.dW1, tb.W1n, tb.b1, tb.db1, tb.b1n], out: ['W1n', 'b1n'],
    cells: cells2('W1n', NJ, NP).concat(cells2('b1n', 1, NJ)),
    focus: (r, c, cell) => {
      if (cell.at[0] === 'b1n') {
        return {
          hl: { b1: [[0, c, 'in']], db1: [[0, c, 'in']], b1n: [[0, c, 'out']] }, sch: {},
          f: sgdText('b₁', 'db₁', c, null, D.b1[c], D.db1[c], D.b1n[c]),
        };
      }
      const i = D.pix[c];
      return {
        hl: { W1: [[r, c, 'in']], dW1: [[r, c, 'in']], W1n: [[r, c, 'out']] },
        sch: { 0: { cell: [r, i] }, 2: { cell: [r, i] }, 4: { cell: [r, i] } },
        f: sgdText('W₁', 'dW₁', r, i, D.W1[r][c], D.dW1[r][c], D.W1n[r][c]),
      };
    },
  },
  {
    phase: '结果', title: '一步更新的效果',
    code: '# 下一轮：x_batch = x[100:200] ... 重复上面所有步骤\n# 每个 epoch 600 步（60000 / 100），默认 12 个 epoch',
    desc: `用更新后的参数对<b>同一个 batch</b> 再做一次前向：真实类别的概率普遍上升，batch 平均 loss 从 <b>${fmt(D.lossMean)}</b> 降到 <b>${fmt(D.lossAfterMean)}</b>。之后取下一个 batch 重复全部步骤，这样的迭代每个 epoch 做 600 次。`,
    schem: [M('X', 100, 784), OP('新参数前向 →'), M("P'", 100, 10)],
    tables: [tb.p, tb.pAfter, tb.lc], out: ['lc'], cells: cells2('lc', NB, 1),
    focus: (b) => {
      const y = D.y[b];
      return {
        hl: { p: [[b, y, 'in']], pAfter: [[b, y, 'out']], lc: [[b, -1, 'out']] },
        sch: { 0: { row: b }, 2: { row: b } },
        f: `样本 ${b}（真实标签 ${y}）\n\n p[${b},${y}]：${fmt(D.p[b][y])} → ${RES(D.pAfter[b][y])}\n` +
          ` loss ：${fmt(D.loss[b])} → ${RES(D.lossAfter[b])}\n 预测 ：${argmax(D.p[b])} → ${argmax(D.pAfter[b])}\n\n` +
          `整个 batch（100 个样本）：\n 平均 loss ${fmt(D.lossMean)} → ${fmt(D.lossAfterMean)}\n 准确率 ${fmt(D.acc * 100)}% → ${fmt(D.accAfter * 100)}%`,
      };
    },
  },
];

const S = (d) => (d <= 1 ? 14 : Math.round(16 + 58 * Math.log10(d)));
function renderSchem(items, hl) {
  let H = 0;
  items.forEach((it) => { if (!it.op) H = Math.max(H, S(it.r)); });
  const top = 26, mid = top + H / 2, out = [];
  let x = 8;
  items.forEach((it, idx) => {
    if (it.op) {
      const w = 16 + it.op.length * 14;
      out.push(`<text x="${x + w / 2}" y="${mid + 6}" class="op">${it.op}</text>`);
      x += w;
      return;
    }
    const w = S(it.c), h = S(it.r), y = mid - h / 2, e = hl[idx];
    out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="mat${e ? ' act' : ''}"/>`);
    if (e) {
      const rh = Math.max(3, h / it.r), cw = Math.max(3, w / it.c);
      const ry = (r) => y + Math.min(h - rh, (r / it.r) * h);
      const cx = (c) => x + Math.min(w - cw, (c / it.c) * w);
      if (e.row != null) out.push(`<rect x="${x}" y="${ry(e.row)}" width="${w}" height="${rh}" class="hlr"/>`);
      if (e.col != null) out.push(`<rect x="${cx(e.col)}" y="${y}" width="${cw}" height="${h}" class="hlr"/>`);
      if (e.cell) out.push(`<rect x="${cx(e.cell[1]) - 1}" y="${ry(e.cell[0]) - 1}" width="${cw + 2}" height="${rh + 2}" class="hlc"/>`);
    }
    out.push(`<text x="${x + w / 2}" y="${y - 7}" class="mn">${it.n}</text>`);
    out.push(`<text x="${x + w / 2}" y="${y + h + 16}" class="ms">${it.r} × ${it.c}</text>`);
    if (it.d) out.push(`<text x="${x + w / 2}" y="${y + h + 31}" class="md">${it.d}</text>`);
    x += w + 8;
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${x + 8} ${top + H + 44}" width="${x + 8}">${out.join('')}</svg>`;
}

function renderTable(t, hl, clickable) {
  const H = hl || [];
  const mark = (r, c) => H.filter(([hr, hc]) => (hr === -1 || hr === r) && (hc === -1 || hc === c)).map((e) => ' ' + e[2]).join('');
  let h = `<div class="tbl"><div class="tt">${t.title}</div><table><tr><th></th>` +
    t.cols.map((c) => `<th>${c}</th>`).join('') + (t.moreC ? '<th></th>' : '') + '</tr>';
  t.data.forEach((row, r) => {
    h += `<tr><th>${t.rows[r]}</th>`;
    row.forEach((v, c) => {
      const cls = ((t.cls && t.cls(r, c, v)) || '') + mark(r, c) + (clickable ? ' click' : '');
      h += `<td class="${cls}" data-t="${t.id}" data-r="${r}" data-c="${c}">${fmt(v)}</td>`;
    });
    if (t.moreC) h += '<td class="more">⋯</td>';
    h += '</tr>';
  });
  if (t.moreR) h += `<tr><th></th><td class="more" colspan="${t.cols.length + (t.moreC ? 1 : 0)}">${t.moreR}</td></tr>`;
  return h + '</table>' + (t.moreC ? `<div class="note">${t.moreC}</div>` : '') + '</div>';
}

if (typeof document !== 'undefined') {
  const $ = (id) => document.getElementById(id);
  let si = 0, ci = 0, timer = null;
  const AUTO = 6;

  function renderImgs(hl) {
    const wrap = document.createElement('div');
    wrap.className = 'imgs';
    for (let b = 0; b < NB; b++) {
      const fig = document.createElement('figure');
      const cv = document.createElement('canvas');
      const s = 5;
      cv.width = 28 * s; cv.height = 28 * s;
      const ctx = cv.getContext('2d');
      D.imgs[b].forEach((v, i) => {
        const g = Math.round(v * 255);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect((i % 28) * s, Math.floor(i / 28) * s, s, s);
      });
      D.pix.forEach((i, c) => {
        const cur = hl.some(([hb, hc]) => hb === b && hc === c);
        ctx.strokeStyle = cur ? '#5b9fd4' : '#e8c54b';
        ctx.lineWidth = cur ? 3 : 1;
        ctx.strokeRect((i % 28) * s + 0.5, Math.floor(i / 28) * s + 0.5, s - 1, s - 1);
      });
      if (hl.some(([hb]) => hb === b)) fig.className = 'sel';
      fig.appendChild(cv);
      const cap = document.createElement('figcaption');
      cap.textContent = `b=${b} · 标签 ${D.y[b]}`;
      fig.appendChild(cap);
      wrap.appendChild(fig);
    }
    return wrap;
  }

  function buildNav() {
    const nav = $('nav');
    let last = '';
    STEPS.forEach((s, idx) => {
      if (s.phase !== last) {
        const h = document.createElement('h2');
        h.textContent = s.phase;
        nav.appendChild(h);
        last = s.phase;
      }
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.dataset.idx = idx;
      btn.textContent = `${idx + 1}. ${s.title}`;
      btn.onclick = () => { stop(); si = idx; ci = 0; show(); };
      nav.appendChild(btn);
    });
  }

  function show() {
    const s = STEPS[si], cell = s.cells[ci], [, r, c] = cell.at;
    const f = s.focus(r, c, cell);
    $('phase').textContent = s.phase;
    $('stitle').textContent = `${si + 1}. ${s.title}`;
    $('code').textContent = s.code;
    $('desc').innerHTML = s.desc;
    $('schem').innerHTML = renderSchem(s.schem, f.sch || {});
    $('tables').innerHTML = s.tables.map((t) => (t.kind === 'img' ? '<div id="imgSlot"></div>' : renderTable(t, f.hl[t.id], s.out.includes(t.id)))).join('');
    const slot = $('imgSlot');
    if (slot) slot.replaceWith(renderImgs(f.hl.img || []));
    $('formula').innerHTML = f.f;
    $('cellInfo').textContent = `第 ${si + 1}/${STEPS.length} 步 · 元素 ${ci + 1}/${s.cells.length}（可点击带虚线下划线的格子）`;
    document.querySelectorAll('nav button').forEach((b) => b.classList.toggle('on', +b.dataset.idx === si));
  }

  function step(d) {
    si = Math.max(0, Math.min(STEPS.length - 1, si + d));
    ci = 0;
    show();
  }
  function cellMove(d) {
    const n = STEPS[si].cells.length;
    if (ci + d >= n) { if (si < STEPS.length - 1) step(1); return; }
    if (ci + d < 0) { if (si > 0) { si--; ci = STEPS[si].cells.length - 1; show(); } return; }
    ci += d;
    show();
  }
  function stop() {
    clearTimeout(timer);
    timer = null;
    $('play').textContent = '播放';
  }
  function tick() {
    const limit = Math.min(AUTO, STEPS[si].cells.length);
    if (ci + 1 < limit) { ci++; show(); }
    else if (si < STEPS.length - 1) step(1);
    else { stop(); return; }
    timer = setTimeout(tick, +$('speed').value);
  }

  $('prevStep').onclick = () => { stop(); step(-1); };
  $('nextStep').onclick = () => { stop(); step(1); };
  $('prevCell').onclick = () => { stop(); cellMove(-1); };
  $('nextCell').onclick = () => { stop(); cellMove(1); };
  $('play').onclick = () => {
    if (timer) { stop(); return; }
    $('play').textContent = '暂停';
    timer = setTimeout(tick, +$('speed').value);
  };
  $('tables').addEventListener('click', (e) => {
    const td = e.target.closest('td.click');
    if (!td) return;
    const t = td.dataset.t, r = +td.dataset.r, c = +td.dataset.c, cells = STEPS[si].cells;
    let idx = cells.findIndex((x) => x.at[0] === t && x.at[1] === r && x.at[2] === c);
    if (idx < 0) idx = cells.findIndex((x) => x.at[0] === t && x.at[1] === r);
    if (idx >= 0) { stop(); ci = idx; show(); }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { stop(); cellMove(1); }
    if (e.key === 'ArrowLeft') { stop(); cellMove(-1); }
  });

  buildNav();
  show();
}
</script>
</body>
</html>
"""


def main():
    data = build_data()
    out = ROOT / "docs" / "train_mnist_step_by_step.html"
    out.write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")
    print(f"batch loss mean {data['lossMean']:.4f} -> {data['lossAfterMean']:.4f}")


if __name__ == "__main__":
    main()

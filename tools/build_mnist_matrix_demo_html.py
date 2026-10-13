#!/usr/bin/env python3
"""Build docs/train_mnist_numpy_matrices.html with embedded tensor JSON."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_NAME = "train_mnist_matrix_demo_data.json"

# (id, label) per group — static sidebar survives slow / failed data load
TENSOR_NAV = [
    ("数据", [("x", "x_batch"), ("y", "y_batch")]),
    ("参数", [
        ("W1", "W₁ (更新前)"),
        ("b1", "b₁ (更新前)"),
        ("W2", "W₂ (更新前)"),
        ("b2", "b₂ (更新前)"),
    ]),
    ("前向", [
        ("z1", "z₁"),
        ("h1", "h₁ = ReLU(z₁)"),
        ("z2", "z₂"),
        ("p", "p = softmax(z₂)"),
    ]),
    ("反向", [("delta", "δ"), ("dz1", "δ_z₁")]),
    ("梯度", [("dW2", "dW₂"), ("db2", "db₂"), ("dW1", "dW₁"), ("db1", "db₁")]),
    ("SGD 后", [("W1a", "W₁ (更新后)")]),
]


def nav_html() -> str:
    parts = []
    for group, items in TENSOR_NAV:
        parts.append(f"<h2>{group}</h2>")
        for tid, label in items:
            parts.append(f'<button type="button" data-id="{tid}">{label}</button>')
    return "\n      ".join(parts)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MNIST MLP — 矩阵逐元素 (B=100)</title>
  <style>
    :root {
      --bg: #0f1117; --panel: #171a22; --stroke: #2a3140;
      --text: #e8eaef; --muted: #8b93a7; --accent: #5b9fd4;
      --pos: #7bc96f; --neg: #d47b9f; --hi: #e8c54b;
      --mono: "IBM Plex Mono", Consolas, monospace;
      --sans: "IBM Plex Sans", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; }
    header { padding: 1rem 1.25rem; border-bottom: 1px solid var(--stroke); }
    header h1 { margin: 0; font-size: 1.15rem; }
    header p { margin: 0.35rem 0 0; color: var(--muted); font-size: 0.85rem; }
    code { font-family: var(--mono); color: var(--accent); font-size: 0.9em; }
    .layout {
      display: grid;
      grid-template-columns: minmax(180px, 220px) minmax(0, 1fr) minmax(260px, 360px);
      grid-template-rows: auto minmax(320px, 1fr);
      height: calc(100vh - 72px);
    }
    @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; grid-template-rows: auto auto auto 1fr; height: auto; } }
    .toolbar {
      grid-column: 1 / -1;
      padding: 0.6rem 1rem;
      border-bottom: 1px solid var(--stroke);
      display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center;
      background: #12151c;
    }
    button, select {
      background: var(--panel); border: 1px solid var(--stroke); color: var(--text);
      padding: 0.35rem 0.65rem; border-radius: 5px; font: inherit; cursor: pointer;
    }
    button.primary { background: var(--accent); color: #0f1117; border-color: var(--accent); font-weight: 600; }
    .tensor-nav { padding: 0.5rem; border-right: 1px solid var(--stroke); overflow-y: auto; }
    .tensor-nav h2 { font-size: 0.7rem; text-transform: uppercase; color: var(--muted); margin: 0.5rem 0 0.35rem; letter-spacing: 0.05em; }
    .tensor-nav button {
      display: block; width: 100%; text-align: left; margin-bottom: 4px; font-size: 0.8rem;
    }
    .tensor-nav button.on { border-color: var(--accent); color: var(--accent); }
    #loadStatus { color: var(--accent); margin: 0.35rem 0 0; font-size: 0.85rem; }
    noscript { color: #e8a54b; display: block; padding: 0.5rem 1rem; }
    .center { padding: 0.5rem; overflow: auto; display: flex; flex-direction: column; gap: 0.5rem; min-height: 400px; }
    .shape-bar { font-family: var(--mono); font-size: 0.75rem; color: var(--muted); }
    .matrix-wrap {
      overflow: auto; flex: 1; border: 1px solid var(--stroke); border-radius: 8px;
      background: var(--panel); position: relative;
    }
    canvas#heatmap { display: block; }
    .grid-table {
      border-collapse: collapse; font-family: var(--mono); font-size: 10px;
    }
    .grid-table th { color: var(--muted); padding: 2px 4px; position: sticky; top: 0; background: var(--panel); }
    .grid-table td {
      padding: 2px 5px; border: 1px solid #1e2430; text-align: right; min-width: 52px;
      cursor: crosshair;
    }
    .grid-table td.hi { background: #2a3548; outline: 1px solid var(--hi); }
    .grid-table td.anim { background: #243328; outline: 1px solid var(--pos); }
    aside {
      border-left: 1px solid var(--stroke); padding: 0.75rem; overflow-y: auto;
      font-size: 0.82rem; line-height: 1.55;
    }
    aside h3 { margin: 0 0 0.5rem; font-size: 0.85rem; }
    .eq { font-family: var(--mono); background: var(--panel); border: 1px solid var(--stroke);
      padding: 0.65rem; border-radius: 6px; margin: 0.5rem 0; font-size: 0.72rem; white-space: pre-wrap; word-break: break-all; }
    .legend { display: flex; align-items: center; gap: 6px; font-size: 0.7rem; color: var(--muted); margin-top: 4px; }
    .legend i { display: inline-block; width: 48px; height: 8px; border-radius: 2px;
      background: linear-gradient(90deg, var(--neg), #333, var(--pos)); }
    .idx-inputs { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; font-size: 0.8rem; }
    .idx-inputs label { color: var(--muted); }
    .idx-inputs input { width: 56px; background: var(--panel); border: 1px solid var(--stroke); color: var(--text); padding: 2px 4px; border-radius: 4px; }
    .progress { flex: 1; min-width: 120px; max-width: 280px; height: 6px; background: var(--stroke); border-radius: 3px; overflow: hidden; }
    .progress > div { height: 100%; background: var(--accent); width: 0%; }
  </style>
  <script>
    window.__setLoadStatus = function (msg) {
      var el = document.getElementById('loadStatus');
      if (el) el.textContent = msg;
    };
    window.__setLoadStatus('HTML 已加载，等待 train_mnist_numpy_matrices.js …');
    window.addEventListener('error', function (e) {
      var m = (e.message || e) + (e.filename ? ' @ ' + e.filename : '');
      window.__setLoadStatus('脚本错误: ' + m);
      var err = document.getElementById('bootErr');
      if (err) { err.hidden = false; err.textContent = m; }
    });
  </script>
</head>
<body>
  <noscript>需要启用 JavaScript。请用 Chrome/Edge 打开 http://127.0.0.1:8765/train_mnist_numpy_matrices.html</noscript>
  <header>
    <h1>train_mnist_numpy · 矩阵与逐元素更新</h1>
    <p>第一个 mini-batch：<code>B = 100</code>，<code>x = train_x[0:100]</code>，与脚本默认 <code>batch_size=100</code>、<code>lr=0.1</code> 一致。点击单元格或播放动画查看每个元素的计算公式与数值。</p>
    <p id="loadStatus">正在初始化…</p>
    <p id="bootErr" hidden style="color:#e8a54b;margin:0.5rem 0 0;font-size:0.85rem;"></p>
  </header>
  <div class="layout">
    <div class="toolbar">
      <button type="button" class="primary" id="btnPlay">播放 SGD 逐元素</button>
      <button type="button" id="btnPause">暂停</button>
      <button type="button" id="btnFwdZ1">演示 z₁[b,j] 求和</button>
      <button type="button" id="btnBwdW1">演示 dW₁[j,i] 求和</button>
      <div class="progress" title="遍历进度"><div id="prog"></div></div>
      <div class="idx-inputs">
        <label>b<input type="number" id="inB" min="0" max="99" value="0" /></label>
        <label>j<input type="number" id="inJ" min="0" max="127" value="0" /></label>
        <label>i<input type="number" id="inI" min="0" max="783" value="0" /></label>
        <label>k<input type="number" id="inK" min="0" max="9" value="7" /></label>
        <label>播放间隔 ms<input type="number" id="inDelay" min="0" max="500" value="0" title="0=最快扫完全部 W₁ 元素" /></label>
      </div>
    </div>
    <nav class="tensor-nav" id="nav">
      __NAV_HTML__
    </nav>
    <section class="center">
      <div class="shape-bar" id="shapeBar"></div>
      <div class="matrix-wrap" id="matrixWrap"></div>
      <div class="legend"><i></i> 负 ← 0 → 正（按当前张量极值缩放）</div>
    </section>
    <aside>
      <h3>当前张量说明</h3>
      <p id="tensorDesc">正在连接数据文件…</p>
      <h3>逐元素公式与数值</h3>
      <div class="eq" id="eqBox">选择左侧张量并点击矩阵中的单元格。</div>
      <h3>SGD 更新（权重/偏置）</h3>
      <div class="eq" id="sgdBox">—</div>
    </aside>
  </div>
"""

HTML_FOOT = r"""
  <script src="train_mnist_numpy_matrices.js" onerror="window.__setLoadStatus('无法加载 train_mnist_numpy_matrices.js — 请用 http://127.0.0.1:8765/ 且服务根目录为 docs')"></script>
</body>
</html>
"""

# The JS is in a separate write to avoid escaping issues - I'll append in the same file write

JS = r"""
let DATA = null;
const DATA_JSON = "__JSON_NAME__";
const DATA_GZ = "__JSON_NAME__.gz";
let loadDetail = '等待加载数据…';

function setStatus(msg) {
  loadDetail = msg;
  if (window.__setLoadStatus) window.__setLoadStatus(msg);
}

const TENSORS = [
  { id: 'x', name: 'x_batch', group: '数据', get: () => DATA.batch.x,
    desc: '输入 mini-batch，行是样本 b∈[0,99]，列是像素 i∈[0,783]。来自 train_x[0:100]，已 ÷255。',
    indices: ['b','i'], formula: (b,i,v) => `x[b=${b}, i=${i}] = ${v}` },
  { id: 'y', name: 'y_batch', group: '数据', vec: () => DATA.labels,
    desc: '整数标签 y_batch[b]，形状 (100,)。',
    indices: ['b'], formula: (b,v) => `y[${b}] = ${v}（类别）` },
  { id: 'W1', name: 'W₁ (更新前)', group: '参数', get: () => DATA.before.W1,
    desc: '第一层权重，形状 (128,784)。前向用 z1[b,j]=Σᵢ x[b,i]·W1[j,i]+b1[j]。',
    indices: ['j','i'], sgd: true, param: 'W1' },
  { id: 'b1', name: 'b₁ (更新前)', group: '参数', vec: () => DATA.before.b1,
    desc: '第一层偏置 (128,)。', indices: ['j'], sgd: true, param: 'b1' },
  { id: 'z1', name: 'z₁', group: '前向', get: () => DATA.batch.z1,
    desc: 'Linear₁ 输出 (100,128)，尚未 ReLU。', indices: ['b','j'],
    formula: (b,j,v) => `z₁[b=${b}, j=${j}] = ${v}\n  = Σᵢ x[b,i]·W₁[j,i] + b₁[j]` },
  { id: 'h1', name: 'h₁ = ReLU(z₁)', group: '前向', get: () => DATA.batch.h1,
    desc: '(100,128)', indices: ['b','j'],
    formula: (b,j,v) => `h₁[b=${b}, j=${j}] = max(0, z₁[b,j]) = ${v}` },
  { id: 'W2', name: 'W₂ (更新前)', group: '参数', get: () => DATA.before.W2,
    desc: '(10,128)', indices: ['k','j'], sgd: true, param: 'W2' },
  { id: 'b2', name: 'b₂ (更新前)', group: '参数', vec: () => DATA.before.b2,
    desc: '(10,)', indices: ['k'], sgd: true, param: 'b2' },
  { id: 'z2', name: 'z₂', group: '前向', get: () => DATA.batch.z2,
    desc: '(100,10) logits', indices: ['b','k'],
    formula: (b,k,v) => `z₂[b=${b}, k=${k}] = Σⱼ h₁[b,j]·W₂[k,j] + b₂[k] = ${v}` },
  { id: 'p', name: 'p = softmax(z₂)', group: '前向', get: () => DATA.batch.p,
    desc: '(100,10) 概率', indices: ['b','k'],
    formula: (b,k,v) => `p[b=${b}, k=${k}] = ${v}` },
  { id: 'delta', name: 'δ', group: '反向', get: () => DATA.batch.delta,
    desc: 'δ = (p − y_onehot) / B，B=100。形状 (100,10)。', indices: ['b','k'],
    formula: (b,k,v) => {
      const y = DATA.labels[b];
      const pb = DATA.batch.p[b][k];
      const oh = k === y ? 1 : 0;
      return `δ[b=${b}, k=${k}] = (p[${b},${k}] − y_onehot[${b},${k}]) / 100\n  = (${pb} − ${oh}) / 100 = ${v}`;
    }},
  { id: 'dW2', name: 'dW₂', group: '梯度', get: () => DATA.grads.dW2,
    desc: 'dW₂ = δ.T @ h₁，形状 (10,128)。', indices: ['k','j'],
    formula: (k,j,v) => `dW₂[k=${k}, j=${j}] = Σ_b δ[b,k]·h₁[b,j] = ${v}` },
  { id: 'db2', name: 'db₂', group: '梯度', vec: () => DATA.grads.db2,
    desc: 'db₂[k] = Σ_b δ[b,k]', indices: ['k'], sgd: true, param: 'b2', grad: 'db2' },
  { id: 'dz1', name: 'δ_z₁', group: '反向', get: () => DATA.batch.delta_z1,
    desc: 'δ_h = δ @ W₂，再乘 ReLU 掩码 (z₁>0)。', indices: ['b','j'] },
  { id: 'dW1', name: 'dW₁', group: '梯度', get: () => DATA.grads.dW1,
    desc: 'dW₁ = δ_z₁.T @ x，形状 (128,784)。', indices: ['j','i'],
    formula: (j,i,v) => `dW₁[j=${j}, i=${i}] = Σ_b δ_z₁[b,j]·x[b,i] = ${v}` },
  { id: 'db1', name: 'db₁', group: '梯度', vec: () => DATA.grads.db1,
    desc: 'db₁[j] = Σ_b δ_z₁[b,j]', indices: ['j'], sgd: true, param: 'b1', grad: 'db1' },
  { id: 'W1a', name: 'W₁ (更新后)', group: 'SGD 后', get: () => DATA.after.W1,
    desc: 'W₁ ← W₁ − 0.1·dW₁', indices: ['j','i'],
    formula: (j,i,v) => {
      const w0 = DATA.before.W1[j][i], g = DATA.grads.dW1[j][i];
      return `W₁_new[j=${j},i=${i}] = W₁ − lr·dW₁\n  = ${w0} − 0.1×(${g}) = ${v}`;
    }},
];

let current = TENSORS[0];
let playing = false, animTimer = null, animMode = null;
let hi = { r: 0, c: 0 };

function $(id) { return document.getElementById(id); }

function wireNav() {
  $('nav').querySelectorAll('button[data-id]').forEach(btn => {
    btn.onclick = () => selectTensor(btn.dataset.id);
  });
}

function selectTensor(id) {
  stopAnim();
  current = TENSORS.find(t => t.id === id);
  if (!current) return;
  $('nav').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.id === id));
  if (!DATA) {
    $('tensorDesc').textContent = loadDetail;
    $('eqBox').textContent = '数据加载完成后会显示矩阵。可先查看左侧张量名称。';
    $('matrixWrap').innerHTML = '<p style="padding:1rem;color:#8b93a7">' + loadDetail + '</p>';
    return;
  }
  $('tensorDesc').textContent = current.desc || '';
  $('matrixWrap').innerHTML = '<p style="padding:1rem;color:#8b93a7">绘制中…</p>';
  requestAnimationFrame(() => {
    renderMatrix();
    focusInputs();
    updateEq();
  });
}

function matrixOf(t) {
  if (t.vec) return t.vec().map(v => [v]);
  return t.get();
}

function shapeStr(M) {
  const r = M.length, c = M[0]?.length ?? 1;
  return `形状 (${r}${c > 1 ? ' × ' + c : ''}) · 共 ${r * c} 个元素`;
}

function valColorRgb(v, vmax) {
  const t = vmax > 0 ? Math.max(-1, Math.min(1, v / vmax)) : 0;
  if (t >= 0) {
    const g = Math.round(40 + t * 80);
    return [Math.round(30 + g * 0.3), g, Math.round(50 + g * 0.2)];
  }
  const r = Math.round(40 + (-t) * 90);
  return [r, Math.round(35 - t * 20), Math.round(55 - t * 15)];
}

function renderMatrix() {
  const M = matrixOf(current);
  const rows = M.length, cols = M[0].length;
  $('shapeBar').textContent = current.name + ' — ' + shapeStr(M) +
    ` · batch 索引 i∈[0,783] 像素 · j∈[0,127] 隐藏 · k∈[0,9] 类别`;

  const wrap = $('matrixWrap');

  if (rows * cols > 8000 || cols > 15) {
    const cv = document.createElement('canvas');
    cv.id = 'heatmap';
    let vmax = 0;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        vmax = Math.max(vmax, Math.abs(M[r][c]));
    cv.width = cols;
    cv.height = rows;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(cols, rows);
    const d = img.data;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [R, G, B] = valColorRgb(M[r][c], vmax);
        const p = (r * cols + c) * 4;
        d[p] = R; d[p + 1] = G; d[p + 2] = B; d[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    cv.style.maxWidth = '100%';
    cv.style.height = 'auto';
    cv.style.imageRendering = 'pixelated';
    cv.onclick = (ev) => {
      const rect = cv.getBoundingClientRect();
      const c = Math.min(cols - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * cols)));
      const r = Math.min(rows - 1, Math.max(0, Math.floor(((ev.clientY - rect.top) / rect.height) * rows)));
      hi = { r, c }; setInputsFromHi(); updateEq();
    };
    wrap.innerHTML = '';
    wrap.appendChild(cv);
    wrap._M = M;
    wrap._mode = 'canvas';
    wrap._img = img;
    wrap._vmax = vmax;
    wrap._rows = rows;
    wrap._cols = cols;
    refreshHiOnly();
    return;
  }

  let thead = '<tr><th></th>';
  for (let c = 0; c < cols; c++) thead += `<th>${c}</th>`;
  thead += '</tr>';
  let body = '';
  for (let r = 0; r < rows; r++) {
    body += `<tr><th>${r}</th>`;
    for (let c = 0; c < cols; c++) {
      const cls = (r === hi.r && c === hi.c) ? ' class="hi"' : '';
      body += `<td${cls} data-r="${r}" data-c="${c}">${fmt(M[r][c])}</td>`;
    }
    body += '</tr>';
  }
  wrap.innerHTML = `<table class="grid-table"><thead>${thead}</thead><tbody>${body}</tbody></table>`;
  wrap._mode = 'table';
  wrap.querySelectorAll('td').forEach(td => {
    td.onclick = () => {
      hi = { r: +td.dataset.r, c: +td.dataset.c };
      wrap.querySelectorAll('td').forEach(x => x.classList.remove('hi', 'anim'));
      td.classList.add('hi');
      setInputsFromHi();
      updateEq();
    };
  });
}

function drawHi(cv, wrap) {
  const ctx = cv.getContext('2d');
  ctx.putImageData(wrap._img, 0, 0);
  ctx.strokeStyle = '#e8c54b';
  ctx.lineWidth = 2;
  ctx.strokeRect(hi.c + 0.5, hi.r + 0.5, 1, 1);
}

function fmt(v) {
  if (typeof v !== 'number') return v;
  if (Math.abs(v) < 1e-4 && v !== 0) return v.toExponential(3);
  return (+v.toFixed(6)).toString();
}

function focusInputs() {
  const idx = current.indices || [];
  $('inB').parentElement.style.display = idx.includes('b') ? '' : 'none';
  $('inJ').parentElement.style.display = idx.includes('j') ? '' : 'none';
  $('inI').parentElement.style.display = idx.includes('i') ? '' : 'none';
  $('inK').parentElement.style.display = idx.includes('k') ? '' : 'none';
}

function setInputsFromHi() {
  const idx = current.indices || [];
  if (idx.includes('b')) $('inB').value = hi.r;
  if (idx.includes('j')) $('inJ').value = idx.length === 2 && idx[0] === 'j' ? hi.r : hi.c;
  if (idx.includes('i')) $('inI').value = hi.c;
  if (idx.includes('k')) {
    if (idx[0] === 'k') $('inK').value = hi.r;
    else $('inK').value = hi.c;
  }
}

function hiFromInputs() {
  const idx = current.indices || [];
  let r = hi.r, c = hi.c;
  if (idx.includes('b')) r = +$('inB').value;
  if (idx.length === 2) {
    if (idx[0] === 'j' && idx[1] === 'i') { r = +$('inJ').value; c = +$('inI').value; }
    else if (idx[0] === 'b' && idx[1] === 'j') { r = +$('inB').value; c = +$('inJ').value; }
    else if (idx[0] === 'b' && idx[1] === 'k') { r = +$('inB').value; c = +$('inK').value; }
    else if (idx[0] === 'k' && idx[1] === 'j') { r = +$('inK').value; c = +$('inJ').value; }
  } else if (idx.length === 1) {
    if (idx[0] === 'j') r = +$('inJ').value;
    if (idx[0] === 'k') r = +$('inK').value;
    if (idx[0] === 'b') r = +$('inB').value;
    c = 0;
  }
  hi = { r, c };
}

function getVal(M, r, c) { return M[r]?.[c] ?? M[r]?.[0]; }

function updateEq() {
  hiFromInputs();
  const M = matrixOf(current);
  const v = getVal(M, hi.r, hi.c);
  const idx = current.indices || [];
  let eq = '';
  if (current.formula) {
    if (idx[0] === 'b' && idx[1] === 'i') eq = current.formula(hi.r, hi.c, v);
    else if (idx[0] === 'b' && idx[1] === 'j') eq = current.formula(hi.r, hi.c, v);
    else if (idx[0] === 'b' && idx[1] === 'k') eq = current.formula(hi.r, hi.c, v);
    else if (idx[0] === 'j' && idx[1] === 'i') eq = current.formula(hi.r, hi.c, v);
    else if (idx[0] === 'k' && idx[1] === 'j') eq = current.formula(hi.r, hi.c, v);
    else if (idx[0] === 'b') eq = current.formula(hi.r, v);
    else if (idx[0] === 'j') eq = `b₁[${hi.r}] = ${v}`;
    else if (idx[0] === 'k') eq = `b₂[${hi.r}] = ${v}`;
  } else {
    eq = `${current.name}[${hi.r},${hi.c}] = ${fmt(v)}`;
  }
  $('eqBox').textContent = eq;

  let sgd = '仅 W₁,W₂,b₁,b₂ 在 backward 后做 SGD 更新。';
  if (current.sgd && current.param) {
    const p = current.param;
    const gradKey = current.grad || ('d' + p);
    const g = DATA.grads[gradKey] || DATA.grads['dW1'];
    let gval, before, after;
    if (p === 'W1') {
      gval = DATA.grads.dW1[hi.r][hi.c];
      before = DATA.before.W1[hi.r][hi.c];
      after = DATA.after.W1[hi.r][hi.c];
      sgd = `W₁[j,i] ← W₁[j,i] − 0.1·dW₁[j,i]\n[j=${hi.r}, i=${hi.c}]\n  ${before} − 0.1×(${gval}) = ${after}`;
    } else if (p === 'W2') {
      gval = DATA.grads.dW2[hi.r][hi.c];
      before = DATA.before.W2[hi.r][hi.c];
      after = DATA.after.W2[hi.r][hi.c];
      sgd = `W₂[k,j] ← W₂[k,j] − 0.1·dW₂[k,j]\n[k=${hi.r}, j=${hi.c}]\n  ${before} − 0.1×(${gval}) = ${after}`;
    } else if (p === 'b1') {
      const j = hi.r;
      sgd = `b₁[${j}] ← ${DATA.before.b1[j]} − 0.1×(${DATA.grads.db1[j]}) = ${DATA.after.b1[j]}`;
    } else if (p === 'b2') {
      const k = hi.r;
      sgd = `b₂[${k}] ← ${DATA.before.b2[k]} − 0.1×(${DATA.grads.db2[k]}) = ${DATA.after.b2[k]}`;
    }
  }
  if (current.id === 'dW1') {
    const j = hi.r, i = hi.c;
    let terms = [], s = 0;
    for (let b = 0; b < 100; b++) {
      const t = DATA.batch.delta_z1[b][j] * DATA.batch.x[b][i];
      s += t;
      if (b < 4) terms.push(`δ_z₁[${b},${j}]·x[${b},${i}]=(${DATA.batch.delta_z1[b][j]})×(${DATA.batch.x[b][i]})`);
    }
    sgd = `dW₁[j,i]=Σ_{b=0}^{99} δ_z₁[b,j]·x[b,i]\n前 4 项 + … = ${fmt(s)}`;
  }
  $('sgdBox').textContent = sgd;
  refreshHiOnly();
}

function refreshHiOnly() {
  const wrap = $('matrixWrap');
  if (wrap._mode === 'canvas' && wrap._img) {
    drawHi(wrap.querySelector('canvas'), wrap);
  } else if (wrap._mode === 'table') {
    wrap.querySelectorAll('td').forEach(td => {
      const r = +td.dataset.r, c = +td.dataset.c;
      td.classList.toggle('hi', r === hi.r && c === hi.c);
    });
  }
}

function stopAnim() {
  playing = false;
  clearTimeout(animTimer);
  animMode = null;
}

function setProg(p) { $('prog').style.width = (p * 100) + '%'; }

$('btnPlay').onclick = () => {
  stopAnim();
  playing = true;
  animMode = 'sgd_W1';
  selectTensor('W1');
  let j = 0, i = 0;
  const total = 128 * 784;
  let step = 0;
  function tick() {
    if (!playing) return;
    hi = { r: j, c: i };
    $('inJ').value = j; $('inI').value = i;
    updateEq();
    step++;
    setProg(step / total);
    i++;
    if (i >= 784) { i = 0; j++; }
    if (j >= 128) { j = 0; i = 0; step = 0; }
    const delay = Math.max(0, +$('inDelay').value || 0);
    animTimer = setTimeout(tick, delay);
  }
  tick();
};

$('btnPause').onclick = stopAnim;

$('btnFwdZ1').onclick = () => {
  stopAnim();
  playing = true;
  selectTensor('z1');
  const b = +$('inB').value, j = +$('inJ').value;
  let i = 0, sum = 0;
  function tick() {
    if (!playing || i > 783) { playing = false; return; }
    const term = DATA.batch.x[b][i] * DATA.before.W1[j][i];
    sum += term;
    $('eqBox').textContent =
      `z₁[b=${b}, j=${j}] = Σᵢ x[b,i]·W₁[j,i] + b₁[j]\n` +
      `累加 i=0..${i}: partial = ${fmt(sum)}` +
      (i === 783 ? `\n+ b₁[${j}]=${DATA.before.b1[j]}\n= ${DATA.batch.z1[b][j]}` : '');
    hi = { r: b, c: j };
    i++;
    setProg(i / 784);
    animTimer = setTimeout(tick, 12);
  }
  tick();
};

$('btnBwdW1').onclick = () => {
  stopAnim();
  playing = true;
  selectTensor('dW1');
  const j = +$('inJ').value, ii = +$('inI').value;
  let b = 0, sum = 0;
  function tick() {
    if (!playing || b > 99) { playing = false; updateEq(); return; }
    sum += DATA.batch.delta_z1[b][j] * DATA.batch.x[b][ii];
    $('eqBox').textContent =
      `dW₁[j=${j}, i=${ii}] = Σ_b δ_z₁[b,j]·x[b,i]\n` +
      `累加 b=0..${b}: partial = ${fmt(sum)}` +
      (b === 99 ? `\n= ${DATA.grads.dW1[j][ii]}` : '');
    hi = { r: j, c: ii };
    b++;
    setProg(b / 100);
    animTimer = setTimeout(tick, 40);
  }
  tick();
};

['inB','inJ','inI','inK'].forEach(id => $(id).onchange = () => updateEq());

function fetchArrayBuffer(url, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url);
    xhr.responseType = 'arraybuffer';
    xhr.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onProgress(e.loaded, e.total, pct);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error('HTTP ' + xhr.status + ' @ ' + url));
    };
    xhr.onerror = () => reject(new Error('网络错误: ' + url));
    xhr.ontimeout = () => reject(new Error('下载超时 (120s): ' + url));
    xhr.timeout = 120000;
    xhr.send();
  });
}

async function decodeJsonBuffer(buf, gz) {
  let bytes = buf;
  if (gz) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('浏览器不支持 gzip 解压，请换 Chrome/Edge/Firefox 新版');
    }
    const ds = new DecompressionStream('gzip');
    bytes = await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer();
  }
  setStatus('正在解析 JSON…');
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function loadData() {
  if (location.protocol === 'file:') {
    throw new Error(
      '不能用 file://。在 docs 目录运行: python3 -m http.server 8765 → http://127.0.0.1:8765/train_mnist_numpy_matrices.html'
    );
  }
  const base = new URL('.', location.href);
  const gzUrl = new URL(DATA_GZ, base).href;
  const jsonUrl = new URL(DATA_JSON, base).href;
  const onProgress = (loaded, total, pct) => {
    const mb = (loaded / (1024 * 1024)).toFixed(1);
    const tmb = total ? (total / (1024 * 1024)).toFixed(1) : '?';
    setStatus(`下载数据 ${mb} / ${tmb} MiB (${pct}%)…`);
  };
  try {
    setStatus('下载 ' + DATA_GZ + ' …');
    const buf = await fetchArrayBuffer(gzUrl, onProgress);
    return await decodeJsonBuffer(buf, true);
  } catch (e1) {
    console.warn(e1);
    setStatus('gzip 失败，尝试 ' + DATA_JSON + ' …');
    const buf = await fetchArrayBuffer(jsonUrl, onProgress);
    return await decodeJsonBuffer(buf, false);
  }
}

async function boot() {
  const err = $('bootErr');
  try {
    DATA = await loadData();
    if (!DATA.batch) throw new Error('JSON 缺少 batch 字段');
    setStatus('加载完成');
    if (err) err.hidden = true;
    selectTensor('x');
  } catch (e) {
    console.error(e);
    setStatus('加载失败');
    if (err) {
      err.hidden = false;
      err.textContent = e.message;
    }
    $('tensorDesc').textContent = '数据未加载。';
    $('eqBox').textContent = String(e.message);
  }
}

function init() {
  wireNav();
  setStatus('准备下载数据…');
  selectTensor('x');
  boot();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
"""

JS_NAME = "train_mnist_numpy_matrices.js"


def main():
    json_path = ROOT / "docs" / JSON_NAME
    if not json_path.exists():
        raise SystemExit(f"Missing {json_path}; run tools/export_mnist_matrix_demo.py first")
    js = JS.replace("__JSON_NAME__", JSON_NAME)
    docs = ROOT / "docs"
    js_out = docs / JS_NAME
    js_out.write_text(js.strip() + "\n")
    body = HTML_TEMPLATE.replace("__NAV_HTML__", nav_html())
    html_out = docs / "train_mnist_numpy_matrices.html"
    html_out.write_text(body + HTML_FOOT)
    print(f"Wrote {html_out} ({html_out.stat().st_size / 1024:.1f} KiB)")
    print(f"Wrote {js_out} ({js_out.stat().st_size / 1024:.1f} KiB)")
    print(f"Data: {json_path} ({json_path.stat().st_size / (1024*1024):.2f} MiB)")


if __name__ == "__main__":
    main()

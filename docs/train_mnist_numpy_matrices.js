let DATA = null;
const DATA_JSON = "train_mnist_matrix_demo_data.json";
const DATA_GZ = "train_mnist_matrix_demo_data.json.gz";
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

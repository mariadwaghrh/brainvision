/* BrainVision — interactive explainer. All computer vision runs live on <canvas>; PR uses exported PCA data. */
(() => {
const D = window.BRAIN_DATA, S = D.samples;
const LEFT = "#5ab8ff", RIGHT = "#ff6f61", SIGNAL = "#d4ff4f", MUTED = "#8b8e9c", GRID = "rgba(255,255,255,.07)", OK = "#6ef0a5", BAD = "#ff6f61";
const N = 128;
const $ = id => document.getElementById(id);
const pct = v => (v * 100).toFixed(1) + "%";

/* ------------------------------------------------------------ i18n */
let lang = new URLSearchParams(location.search).get("lang") || localStorage.getItem("bv-lang") || "en";
if (!window.I18N[lang]) lang = "en";
const t = (k, v = {}) => (window.I18N[lang][k] ?? window.I18N.en[k] ?? k).replace(/\{(\w+)\}/g, (_, x) => v[x]);
const labelName = l => t(l ? "right" : "left");
const rerenders = [];
function applyLang() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  document.querySelectorAll("[data-t]").forEach(el => { el.innerHTML = t(el.dataset.t); });
  $("langBtn").textContent = lang === "ar" ? "EN" : "ع";
  rerenders.forEach(f => f());
}
$("langBtn").onclick = () => { lang = lang === "ar" ? "en" : "ar"; localStorage.setItem("bv-lang", lang); applyLang(); };

/* ------------------------------------------------------------ image helpers */
const imgCache = new Map();
function loadImg(s) {
  const key = s.label + s.name;
  if (!imgCache.has(key)) imgCache.set(key, new Promise(res => { const im = new Image(); im.onload = () => res(im); im.src = s.img; }));
  return imgCache.get(key);
}
function toPixels(im) {
  const c = document.createElement("canvas"); c.width = c.height = N;
  const x = c.getContext("2d"); x.fillStyle = "#fff"; x.fillRect(0, 0, N, N); x.drawImage(im, 0, 0, N, N);
  return x.getImageData(0, 0, N, N).data;
}
function putRGB(canvas, fn) {
  const x = canvas.getContext("2d"), w = canvas.width, h = canvas.height, id = x.createImageData(w, h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) { const c = fn(i); id.data[p] = c[0]; id.data[p + 1] = c[1]; id.data[p + 2] = c[2]; id.data[p + 3] = 255; }
  x.putImageData(id, 0, 0);
}
const offscreen = fn => { const c = document.createElement("canvas"); c.width = c.height = N; putRGB(c, fn); return c; };
function circleMask(r) {
  const m = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) m[y * N + x] = (x - 64) ** 2 + (y - 64) ** 2 <= r * r ? 1 : 0;
  return m;
}
function rgb2hsv(r, g, b) { // OpenCV 8-bit: H 0..180, S,V 0..255
  const v = Math.max(r, g, b), d = v - Math.min(r, g, b), s = v ? (d * 255) / v : 0;
  let h = 0;
  if (d) { h = v === r ? (60 * (g - b)) / d : v === g ? 120 + (60 * (b - r)) / d : 240 + (60 * (r - g)) / d; if (h < 0) h += 360; }
  return [h / 2, s, v];
}
function hue2rgb(h) {
  const H = h * 2, X = 1 - Math.abs(((H / 60) % 2) - 1);
  return [[1, X, 0], [X, 1, 0], [0, 1, X], [0, X, 1], [X, 0, 1], [1, 0, X]][Math.floor(H / 60) % 6].map(v => v * 255);
}
const clamp01 = v => Math.max(0, Math.min(1, v));
const jet = t => [clamp01(1.5 - Math.abs(4 * t - 3)) * 255, clamp01(1.5 - Math.abs(4 * t - 2)) * 255, clamp01(1.5 - Math.abs(4 * t - 1)) * 255];
const JET = Array.from({ length: 256 }, (_, i) => jet(i / 255));

function analyse(px, r) {
  const mask = circleMask(r), hsv = new Float32Array(N * N * 3), gray = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const R = mask[i] ? px[i * 4] : 0, G = mask[i] ? px[i * 4 + 1] : 0, B = mask[i] ? px[i * 4 + 2] : 0;
    hsv.set(rgb2hsv(R, G, B), i * 3); gray[i] = 0.299 * R + 0.587 * G + 0.114 * B;
  }
  return { mask, hsv, gray };
}
function recoverActivity(px, mask) { // invert the jet colormap
  const val = new Float32Array(N * N).fill(NaN);
  for (let i = 0; i < N * N; i++) {
    if (!mask[i]) continue;
    const r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2];
    let best = 1e9, bi = 0;
    for (let k = 0; k < 256; k++) { const c = JET[k], d = (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2; if (d < best) { best = d; bi = k; } }
    if (best < 1600) val[i] = bi / 255;
  }
  return val;
}
function activityRGB(act, stretch) {
  let lo = 0, hi = 1;
  if (stretch) {
    const v = Array.from(act).filter(x => !isNaN(x)).sort((a, b) => a - b);
    if (v.length) { lo = v[Math.floor(v.length * .02)]; hi = v[Math.floor(v.length * .98)]; }
    if (hi - lo < 1e-6) hi = lo + 1e-6;
  }
  return i => (isNaN(act[i]) ? [13, 15, 22] : jet(clamp01((act[i] - lo) / (hi - lo))));
}
function asymmetry(act) {
  let sl = 0, nl = 0, sr = 0, nr = 0; const all = [];
  for (let i = 0; i < N * N; i++) { const v = act[i]; if (isNaN(v)) continue; all.push(v); if (i % N < 64) { sl += v; nl++; } else { sr += v; nr++; } }
  if (!nl || !nr) return 0;
  const m = all.reduce((a, b) => a + b, 0) / all.length, sd = Math.sqrt(all.reduce((a, b) => a + (b - m) ** 2, 0) / all.length) || 1;
  return (sr / nr - sl / nl) / sd;
}
/* The two features the final model relies on (same as 2_cv_feature_extraction.py) */
function featureMaps(act, mask) {
  // 16x16 activity grid; pixels off the colormap count as 0.5 (average activity)
  const filled = Array.from(act, v => (isNaN(v) ? .5 : v)), grid = new Float32Array(256);
  for (let i = 0; i < N * N; i++) grid[((i / N) >> 3) * 16 + ((i % N) >> 3)] += filled[i] / 64;
  // mirror asymmetry: each pixel minus its mirror image across the midline
  let mm = 1e-6;
  const mir = filled.map((v, i) => { const d = v - filled[i - (i % N) + (N - 1 - (i % N))]; mm = Math.max(mm, Math.abs(d)); return d; });
  return {
    grid: i => (mask[i] ? jet(grid[((i / N) >> 3) * 16 + ((i % N) >> 3)]) : [13, 15, 22]),
    mirror: i => { if (!mask[i]) return [13, 15, 22]; const r = mir[i] / mm, d = Math.sign(r) * Math.sqrt(Math.abs(r));
      return d >= 0 ? [30 + 225 * d, 30 + 81 * d, 40 + 57 * d] : [30 + 60 * -d, 30 + 154 * -d, 40 + 215 * -d]; },
  };
}
function canny(gray, lo, hi) { // 3x3 Sobel, L1 magnitude, NMS, hysteresis (OpenCV defaults)
  const gx = new Float32Array(N * N), gy = new Float32Array(N * N), mag = new Float32Array(N * N);
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
    const p = (dx, dy) => gray[(y + dy) * N + x + dx], i = y * N + x;
    gx[i] = -p(-1, -1) - 2 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2 * p(1, 0) + p(1, 1);
    gy[i] = -p(-1, -1) - 2 * p(0, -1) - p(1, -1) + p(-1, 1) + 2 * p(0, 1) + p(1, 1);
    mag[i] = Math.abs(gx[i]) + Math.abs(gy[i]);
  }
  const nms = new Float32Array(N * N);
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
    const i = y * N + x, m = mag[i]; if (!m) continue;
    let a = (Math.atan2(gy[i], gx[i]) * 180) / Math.PI; if (a < 0) a += 180;
    const [o1, o2] = a < 22.5 || a >= 157.5 ? [-1, 1] : a < 67.5 ? [-N - 1, N + 1] : a < 112.5 ? [-N, N] : [-N + 1, N - 1];
    if (m >= mag[i + o1] && m >= mag[i + o2]) nms[i] = m;
  }
  const out = new Uint8Array(N * N), st = [];
  for (let i = 0; i < N * N; i++) if (nms[i] > hi) { out[i] = 255; st.push(i); }
  while (st.length) { const i = st.pop(); for (const d of [-N - 1, -N, -N + 1, -1, 1, N - 1, N, N + 1]) { const j = i + d; if (j >= 0 && j < N * N && !out[j] && nms[j] > lo) { out[j] = 255; st.push(j); } } }
  return out;
}
function hogCells(gray) {
  const bins = new Float32Array(64 * 9);
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
    const dx = gray[y * N + x + 1] - gray[y * N + x - 1], dy = gray[(y + 1) * N + x] - gray[(y - 1) * N + x], m = Math.hypot(dx, dy);
    if (!m) continue;
    let a = (Math.atan2(dy, dx) * 180) / Math.PI; if (a < 0) a += 180;
    bins[((y >> 4) * 8 + (x >> 4)) * 9 + Math.min(8, Math.floor(a / 20))] += m;
  }
  return bins;
}
function drawHog(ctx, ox, oy, W, gray, under) {
  const cs = W / 8;
  ctx.save(); ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#000"; ctx.fillRect(ox, oy, W, W);
  if (under) { ctx.globalAlpha = .45; ctx.drawImage(under, ox, oy, W, W); ctx.globalAlpha = 1; }
  const bins = hogCells(gray), mx = Math.max(...bins) || 1;
  ctx.lineCap = "round";
  for (let cy = 0; cy < 8; cy++) for (let cx = 0; cx < 8; cx++) for (let b = 0; b < 9; b++) {
    const v = bins[(cy * 8 + cx) * 9 + b] / mx; if (v < .03) continue;
    const ang = ((b * 20 + 100) * Math.PI) / 180, len = (cs / 2 - 2) * Math.min(1, v * 1.4);
    const x0 = ox + cx * cs + cs / 2, y0 = oy + cy * cs + cs / 2;
    ctx.strokeStyle = `rgba(212,255,79,${Math.min(1, .25 + v)})`; ctx.lineWidth = 1 + v * 2 * (W / 256);
    ctx.beginPath(); ctx.moveTo(x0 - Math.cos(ang) * len, y0 - Math.sin(ang) * len); ctx.lineTo(x0 + Math.cos(ang) * len, y0 + Math.sin(ang) * len); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,.07)"; ctx.lineWidth = 1;
  for (let k = 1; k < 8; k++) { ctx.beginPath(); ctx.moveTo(ox + k * cs, oy); ctx.lineTo(ox + k * cs, oy + W); ctx.moveTo(ox, oy + k * cs); ctx.lineTo(ox + W, oy + k * cs); ctx.stroke(); }
  ctx.restore();
}
function hsHist(hsv, mask, x0, x1) {
  const h = new Float32Array(64);
  for (let y = 0; y < N; y++) for (let x = x0; x < x1; x++) {
    const i = y * N + x; if (!mask[i]) continue;
    h[Math.min(7, Math.floor(hsv[i * 3] / 22.5)) * 8 + Math.min(7, Math.floor(hsv[i * 3 + 1] / 32))]++;
  }
  const n = Math.hypot(...h) || 1; return h.map(v => v / n);
}
function drawHist(canvas, h, diverging) {
  const x = canvas.getContext("2d"), c = canvas.width / 8, mx = Math.max(1e-6, ...h.map(Math.abs));
  x.fillStyle = "#0b0d13"; x.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < 64; i++) {
    const v = h[i] / mx;
    x.fillStyle = diverging ? (v >= 0 ? `rgba(255,111,97,${v})` : `rgba(90,184,255,${-v})`) : `rgba(79,227,193,${Math.sqrt(Math.max(0, v))})`;
    x.beginPath(); x.roundRect((i % 8) * c + 1.5, Math.floor(i / 8) * c + 1.5, c - 3, c - 3, 3); x.fill();
  }
}

/* ------------------------------------------------------------ chart helpers */
function axes(ctx, W, H, pad, xr, yr, xl, yl) {
  ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.font = "10px JetBrains Mono, monospace"; ctx.lineWidth = 1;
  for (let k = 0; k <= 5; k++) {
    const gx = pad.l + ((W - pad.l - pad.r) * k) / 5, gy = pad.t + ((H - pad.t - pad.b) * k) / 5;
    ctx.beginPath(); ctx.moveTo(gx, pad.t); ctx.lineTo(gx, H - pad.b); ctx.moveTo(pad.l, gy); ctx.lineTo(W - pad.r, gy); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText((xr[0] + ((xr[1] - xr[0]) * k) / 5).toFixed(0), gx, H - pad.b + 14);
    ctx.textAlign = "right"; ctx.fillText((yr[1] - ((yr[1] - yr[0]) * k) / 5).toFixed(0), pad.l - 6, gy + 3);
  }
  if (xl) { ctx.textAlign = "center"; ctx.fillText(xl, (W + pad.l - pad.r) / 2, H - 4); }
  if (yl) { ctx.save(); ctx.translate(11, (H + pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yl, 0, 0); ctx.restore(); }
}
const range = (a, m = .08) => { const lo = Math.min(...a), hi = Math.max(...a), d = (hi - lo) * m || 1; return [lo - d, hi + d]; };
function marker(ctx, x, y, r, diamond) {
  ctx.beginPath();
  if (diamond) { ctx.moveTo(x, y - r * 1.3); ctx.lineTo(x + r * 1.3, y); ctx.lineTo(x, y + r * 1.3); ctx.lineTo(x - r * 1.3, y); ctx.closePath(); }
  else ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
}
function rng(a) { return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function split(seed, testPct) {
  const r = rng(seed), train = [], test = [];
  for (const lab of [0, 1]) {
    const idx = S.map((_, i) => i).filter(i => S[i].label === lab);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const nt = Math.round((idx.length * testPct) / 100); test.push(...idx.slice(0, nt)); train.push(...idx.slice(nt));
  }
  return { train, test };
}
function knn(q, train, k, d) {
  const ds = train.map(i => { let s = 0; for (let j = 0; j < d; j++) s += (S[i].pca[j] - q[j]) ** 2; return [s, S[i].label]; }).sort((a, b) => a[0] - b[0]);
  const kk = Math.min(k, ds.length); let v = 0; for (let j = 0; j < kk; j++) v += ds[j][1];
  return v * 2 > kk ? 1 : 0;
}
function decisionMap(ctx, W, H, pad, xr, yr, train, k, d) {
  const step = 6;
  for (let py = pad.t; py < H - pad.b; py += step) for (let px = pad.l; px < W - pad.r; px += step) {
    const q = [xr[0] + ((px + step / 2 - pad.l) / (W - pad.l - pad.r)) * (xr[1] - xr[0]), yr[1] - ((py + step / 2 - pad.t) / (H - pad.t - pad.b)) * (yr[1] - yr[0])];
    ctx.fillStyle = knn(q, train, k, d) ? "rgba(255,111,97,.13)" : "rgba(90,184,255,.13)";
    ctx.fillRect(px, py, step, step);
  }
}

/* ------------------------------------------------------------ hero: live EEG traces */
(function heroEEG() {
  const c = $("eeg"), ctx = c.getContext("2d"); let W, H, t0 = 0, running = true;
  const CH = 14, chans = Array.from({ length: CH }, (_, i) => ({
    f: [8 + Math.random() * 5, 14 + Math.random() * 10, 22 + Math.random() * 8], a: [1, .5 + Math.random() * .4, .3], ph: Math.random() * 100,
    col: i === 4 ? LEFT : i === 9 ? RIGHT : null,
  }));
  const fit = () => { const dpr = Math.min(2, devicePixelRatio || 1); W = c.clientWidth; H = c.clientHeight; c.width = W * dpr; c.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
  fit(); addEventListener("resize", fit);
  new IntersectionObserver(e => { running = e[0].isIntersecting; if (running) requestAnimationFrame(draw); }).observe(c);
  function draw(ts) {
    if (!running) return;
    t0 = ts / 1000; ctx.clearRect(0, 0, W, H);
    const gap = H / (CH + 1);
    chans.forEach((ch, i) => {
      const y0 = gap * (i + 1);
      ctx.beginPath();
      for (let x = 0; x <= W; x += 3) {
        const tt = x / 260 - t0 * .35 + ch.ph;
        const burst = .6 + .4 * Math.sin(tt * .7 + i);
        const v = (ch.a[0] * Math.sin(tt * ch.f[0]) + ch.a[1] * Math.sin(tt * ch.f[1] * 1.3) + ch.a[2] * Math.sin(tt * ch.f[2] * 2.1)) * burst;
        x ? ctx.lineTo(x, y0 + v * gap * .28) : ctx.moveTo(x, y0 + v * gap * .28);
      }
      ctx.strokeStyle = ch.col || "rgba(236,233,226,.22)"; ctx.lineWidth = ch.col ? 1.6 : 1; ctx.stroke();
    });
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();

function heroStats() {
  $("heroStats").innerHTML = [[S.length, "stat.images"], [D.nFeatures.toLocaleString("en"), "stat.features"], [D.explainedVariance.length, "stat.pcs"], [D.svm.cvAcc + "%", "stat.cv", "warn"]]
    .map(([b, k, c]) => `<div class="${c || ""}"><b dir="ltr">${b}</b><span>${t(k)}</span></div>`).join("");
}
rerenders.push(heroStats);

/* ------------------------------------------------------------ story: scrollytelling stage */
const storyIdx = (() => { // most confident correct test-set prediction
  let best = -1, bc = 0;
  S.forEach((s, i) => { if (s.split !== "test") return; const p = s.svmProb, ok = (p > .5 ? 1 : 0) === s.label, c = Math.abs(p - .5); if (ok && c > bc) { bc = c; best = i; } });
  return best >= 0 ? best : 0;
})();
const stage = $("stage"), sctx = stage.getContext("2d"), SW = stage.width;
stage.style.transition = "opacity .25s";
let stageStep = -1, stagePx = null, stageAnim = 0;
const STAGE_LABELS = ["signal → 8–30 Hz", "topomap · 128×128", "mask · HSV · activity · mirror", "PCA · PC1 × PC2", "k-NN decision region", "prediction"];
$("stageDots").innerHTML = STAGE_LABELS.map(() => "<i></i>").join("");

function stageEEG(ts) {
  if (stageStep !== 0) return;
  sctx.fillStyle = "#0b0d13"; sctx.fillRect(0, 0, SW, SW);
  const names = ["FC3", "C3", "CP3", "Cz", "CP4", "C4", "FC4", "Pz"], gap = SW / (names.length + 1), tt0 = ts / 1000;
  names.forEach((n, i) => {
    const y0 = gap * (i + 1), col = n === "C3" ? LEFT : n === "C4" ? RIGHT : "rgba(236,233,226,.45)";
    sctx.fillStyle = MUTED; sctx.font = "11px JetBrains Mono, monospace"; sctx.textAlign = "left"; sctx.fillText(n, 14, y0 + 4);
    sctx.beginPath();
    for (let x = 56; x < SW - 14; x += 2) {
      const tt = x / 90 - tt0 * .8 + i * 3.1;
      const v = Math.sin(tt * 10) * .6 + Math.sin(tt * 23 + i) * .3 + Math.sin(tt * 3.3) * .25;
      x === 56 ? sctx.moveTo(x, y0 + v * gap * .32) : sctx.lineTo(x, y0 + v * gap * .32);
    }
    sctx.strokeStyle = col; sctx.lineWidth = col.startsWith("#") ? 1.8 : 1; sctx.stroke();
  });
  stageAnim = requestAnimationFrame(stageEEG);
}
async function renderStage(i) {
  if (i === stageStep) return;
  stageStep = i; cancelAnimationFrame(stageAnim);
  document.querySelectorAll("#stageDots i").forEach((d, k) => d.classList.toggle("on", k <= i));
  $("stageLabel").textContent = `0${i + 1} · ${STAGE_LABELS[i]}`;
  stage.style.opacity = 0;
  await new Promise(r => setTimeout(r, 200));
  if (stageStep !== i) return;
  if (!stagePx) stagePx = toPixels(await loadImg(S[storyIdx]));
  const s = S[storyIdx], px = stagePx;
  sctx.save(); sctx.clearRect(0, 0, SW, SW); sctx.fillStyle = "#0b0d13"; sctx.fillRect(0, 0, SW, SW); sctx.imageSmoothingEnabled = false;
  if (i === 0) { sctx.restore(); stage.style.opacity = 1; stageAnim = requestAnimationFrame(stageEEG); return; }
  if (i === 1) {
    sctx.fillStyle = "#fff"; sctx.beginPath(); sctx.roundRect(60, 60, SW - 120, SW - 120, 22); sctx.fill();
    sctx.imageSmoothingEnabled = true; sctx.drawImage(await loadImg(s), 80, 80, SW - 160, SW - 160);
  }
  if (i === 2) {
    const { mask, hsv } = analyse(px, 54), g = 14, w = (SW - g * 3) / 2;
    const tiles = [
      offscreen(k => (mask[k] ? [px[k * 4], px[k * 4 + 1], px[k * 4 + 2]] : [0, 0, 0])),
      offscreen(k => ((k % N) === 64 && Math.floor(k / N) % 6 < 3 ? [255, 255, 255] : hue2rgb(hsv[k * 3]))),
    ];
    const fm = featureMaps(recoverActivity(px, mask), mask);
    tiles.push(offscreen(fm.grid), offscreen(fm.mirror));
    tiles.forEach((c, k) => { const x = g + (k % 2) * (w + g), y = g + Math.floor(k / 2) * (w + g); sctx.drawImage(c, x, y, w, w); });
    sctx.fillStyle = "rgba(0,0,0,.6)"; sctx.font = "11px JetBrains Mono, monospace"; sctx.textAlign = "left";
    ["ROI", "HSV · H", "ACTIVITY 16×16", "MIRROR L−R"].forEach((l, k) => { const x = g + (k % 2) * (w + g), y = g + Math.floor(k / 2) * (w + g);
      sctx.fillStyle = "rgba(0,0,0,.65)"; sctx.fillRect(x + 8, y + 8, l.length * 7.5 + 12, 20); sctx.fillStyle = "#fff"; sctx.fillText(l, x + 14, y + 22); });
  }
  if (i === 3 || i === 4) {
    const pad = { l: 40, r: 16, t: 16, b: 32 }, xr = range(S.map(q => q.pca[0])), yr = range(S.map(q => q.pca[1]));
    const sx = v => pad.l + ((v - xr[0]) / (xr[1] - xr[0])) * (SW - pad.l - pad.r), sy = v => SW - pad.b - ((v - yr[0]) / (yr[1] - yr[0])) * (SW - pad.t - pad.b);
    const tr = S.map((_, k) => k).filter(k => S[k].split === "train");
    if (i === 4) decisionMap(sctx, SW, SW, pad, xr, yr, tr, 7, 2);
    axes(sctx, SW, SW, pad, xr, yr, "PC1", "PC2");
    S.forEach((q, k) => { if (k === storyIdx) return; sctx.fillStyle = (q.label ? RIGHT : LEFT) + (i === 4 && q.split === "test" ? "33" : "bb"); sctx.strokeStyle = "#0b0d13"; sctx.lineWidth = 1; marker(sctx, sx(q.pca[0]), sy(q.pca[1]), 4.5, false); });
    const X = sx(s.pca[0]), Y = sy(s.pca[1]);
    sctx.strokeStyle = SIGNAL; sctx.lineWidth = 2; sctx.beginPath(); sctx.arc(X, Y, 14, 0, Math.PI * 2); sctx.stroke();
    sctx.fillStyle = SIGNAL; marker(sctx, X, Y, 5, true);
  }
  if (i === 5) {
    const p = s.svmProb, pred = p > .5 ? 1 : 0, col = pred ? RIGHT : LEFT;
    sctx.fillStyle = "#fff"; sctx.beginPath(); sctx.roundRect(SW / 2 - 110, 50, 220, 220, 18); sctx.fill();
    sctx.imageSmoothingEnabled = true; sctx.drawImage(await loadImg(s), SW / 2 - 98, 62, 196, 196);
    sctx.textAlign = "center"; sctx.fillStyle = MUTED; sctx.font = "12px JetBrains Mono, monospace";
    sctx.fillText(`SVM PREDICTS · ${pred ? "RIGHT" : "LEFT"} HAND`, SW / 2, 316);
    sctx.fillStyle = col; sctx.font = "400 64px Fraunces, Georgia, serif"; sctx.fillText(pct(Math.max(p, 1 - p)), SW / 2, 384);
    const bx = 70, bw = SW - 140, by = 412;
    sctx.fillStyle = LEFT; sctx.beginPath(); sctx.roundRect(bx, by, bw, 8, 4); sctx.fill();
    sctx.fillStyle = RIGHT; sctx.beginPath(); sctx.roundRect(bx + bw * (1 - p), by, bw * p, 8, 4); sctx.fill();
    sctx.fillStyle = "#fff"; sctx.fillRect(bx + bw / 2 - 1, by - 6, 2, 20);
    sctx.font = "11px JetBrains Mono, monospace"; sctx.fillStyle = LEFT; sctx.textAlign = "left"; sctx.fillText("LEFT", bx, by + 30);
    sctx.fillStyle = RIGHT; sctx.textAlign = "right"; sctx.fillText("RIGHT", bx + bw, by + 30);
    const ok = pred === s.label;
    sctx.textAlign = "center"; sctx.fillStyle = ok ? OK : BAD; sctx.font = "13px JetBrains Mono, monospace";
    sctx.fillText(`${ok ? "✓ CORRECT" : "✗ WRONG"} · TRUE: ${s.label ? "RIGHT" : "LEFT"} · ${s.name}`, SW / 2, 486);
  }
  sctx.restore(); stage.style.opacity = 1;
}
const stepObs = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { document.querySelectorAll(".step").forEach(s => s.classList.toggle("active", s === e.target)); renderStage(+e.target.dataset.step); }
}), { rootMargin: "-45% 0px -45% 0px" });
document.querySelectorAll(".step").forEach(s => stepObs.observe(s));
renderStage(0);

/* ------------------------------------------------------------ vision lab */
let cur = 0, filter = "all", hsvCh = 0, px = null;
const gal = $("gallery");
S.forEach((s, i) => { const im = document.createElement("img"); im.src = s.img; im.className = s.label ? "r" : "l"; im.dataset.i = i; im.onclick = () => selectSample(i); gal.appendChild(im); });
document.querySelectorAll("#classFilter button").forEach(b => b.onclick = () => {
  filter = b.dataset.c; document.querySelectorAll("#classFilter button").forEach(o => o.classList.toggle("on", o === b));
  gal.querySelectorAll("img").forEach(im => (im.style.display = filter === "all" || S[im.dataset.i].label === +filter ? "" : "none"));
});
$("randomBtn").onclick = () => { const pool = S.map((_, i) => i).filter(i => filter === "all" || S[i].label === +filter); selectSample(pool[Math.floor(Math.random() * pool.length)]); };
document.querySelectorAll("#hsvCh button").forEach(b => b.onclick = () => { hsvCh = +b.dataset.ch; document.querySelectorAll("#hsvCh button").forEach(o => o.classList.toggle("on", o === b)); runCV(); });
for (const id of ["radius", "cLow", "cHigh"]) $(id).oninput = () => { $(id + "V").textContent = $(id).value; runCV(); };
$("hogOverlay").onchange = runCV; $("jetStretch").onchange = runCV;

function currentName() { if (S[cur]) $("currentName").innerHTML = `${S[cur].name} · <span class="${S[cur].label ? "c-right" : "c-left"}">${labelName(S[cur].label)}</span>`; }
async function selectSample(i) {
  cur = i;
  gal.querySelectorAll("img").forEach(im => im.classList.toggle("sel", +im.dataset.i === i));
  const sel = gal.querySelector(`img[data-i="${i}"]`); if (sel) gal.scrollTo({ left: sel.offsetLeft - gal.clientWidth / 2, behavior: "smooth" });
  currentName();
  px = toPixels(await loadImg(S[i])); runCV();
}
function runCV() {
  if (!px) return;
  const { mask, hsv, gray } = analyse(px, +$("radius").value);
  putRGB($("cvRaw"), i => [px[i * 4], px[i * 4 + 1], px[i * 4 + 2]]);
  const masked = i => (mask[i] ? [px[i * 4], px[i * 4 + 1], px[i * 4 + 2]] : [0, 0, 0]);
  putRGB($("cvMask"), masked);
  putRGB($("cvHsv"), i => { if ((i % N) === 64 && Math.floor(i / N) % 6 < 3) return [255, 255, 255]; const v = hsv[i * 3 + hsvCh]; return hsvCh ? [v, v, v] : hue2rgb(v); });
  const e = canny(gray, +$("cLow").value, +$("cHigh").value); let ne = 0; e.forEach(v => (ne += v ? 1 : 0));
  putRGB($("cvEdge"), i => (e[i] ? [212, 255, 79] : [0, 0, 0]));
  $("edgeCount").textContent = ne ? t("cv.edges.some", { n: ne }) : t("cv.edges.none");
  drawHog($("cvHog").getContext("2d"), 0, 0, $("cvHog").width, gray, $("hogOverlay").checked ? offscreen(masked) : null);
  const act = recoverActivity(px, mask); putRGB($("cvJet"), activityRGB(act, $("jetStretch").checked));
  const fm = featureMaps(act, mask);
  putRGB($("cvGrid"), fm.grid); putRGB($("cvMirror"), fm.mirror);
  const hl = hsHist(hsv, mask, 0, 64), hr = hsHist(hsv, mask, 64, N);
  drawHist($("hLeft"), hl); drawHist($("hRight"), hr); drawHist($("hDiff"), hl.map((v, k) => v - hr[k]), true);
  const a = asymmetry(act);
  $("asymVal").textContent = (a >= 0 ? "+" : "") + a.toFixed(2); $("asymVal").style.color = a >= 0 ? RIGHT : LEFT;
  $("asymMarker").style.left = `calc(${(clamp01((a + 1) / 2)) * 100}% - 1.5px)`;
}
rerenders.push(currentName, runCV);

/* ------------------------------------------------------------ pattern lab: scatter */
const NPC = D.explainedVariance.length;
for (const sel of [$("pcX"), $("pcY")]) for (let k = 0; k < NPC; k++) sel.add(new Option(`PC${k + 1}`, k));
$("pcY").value = 1;
let colorBy = "label", hover = -1, selected = -1, pts = [];
$("pcX").onchange = $("pcY").onchange = drawScatter;
$("colorBy").onchange = () => { colorBy = $("colorBy").value; drawScatter(); };
function pointColor(s) {
  const pred = s.cvSvmPred;
  if (colorBy === "label") return s.label ? RIGHT : LEFT;
  if (colorBy === "svm") return pred ? RIGHT : LEFT;
  return pred === s.label ? OK : BAD;
}
function drawScatter() {
  const c = $("scatter"), ctx = c.getContext("2d"), W = c.width, H = c.height, pad = { l: 44, r: 14, t: 12, b: 34 };
  const xi = +$("pcX").value, yi = +$("pcY").value, xr = range(S.map(s => s.pca[xi])), yr = range(S.map(s => s.pca[yi]));
  ctx.clearRect(0, 0, W, H);
  axes(ctx, W, H, pad, xr, yr, `PC${xi + 1} · ${pct(D.explainedVariance[xi])}`, `PC${yi + 1} · ${pct(D.explainedVariance[yi])}`);
  const sx = v => pad.l + ((v - xr[0]) / (xr[1] - xr[0])) * (W - pad.l - pad.r), sy = v => H - pad.b - ((v - yr[0]) / (yr[1] - yr[0])) * (H - pad.t - pad.b);
  pts = S.map(s => [sx(s.pca[xi]), sy(s.pca[yi])]);
  S.forEach((s, i) => {
    const hi = i === hover || i === selected;
    ctx.fillStyle = pointColor(s) + (hi ? "" : "bb"); ctx.strokeStyle = i === selected ? "#fff" : i === hover ? SIGNAL : "#10131c"; ctx.lineWidth = hi ? 2.5 : 1;
    marker(ctx, pts[i][0], pts[i][1], hi ? 7 : 5, s.split === "test");
  });
  const items = colorBy === "correct" ? [[OK, "lg.ok"], [BAD, "lg.no"]] : colorBy === "svm" ? [[LEFT, "lg.pl"], [RIGHT, "lg.pr"]] : [[LEFT, "lg.left"], [RIGHT, "lg.right"]];
  $("scLegend").innerHTML = items.map(([c, k]) => `<span><i style="background:${c}"></i>${t(k)}</span>`).join("") + `<span>${t("lg.test")}</span>`;
}
function nearest(e, canvas, list, maxD = 14) {
  const r = canvas.getBoundingClientRect(), mx = ((e.clientX - r.left) * canvas.width) / r.width, my = ((e.clientY - r.top) * canvas.height) / r.height;
  let best = -1, bd = maxD * maxD; list.forEach(([x, y], i) => { const d = (x - mx) ** 2 + (y - my) ** 2; if (d < bd) { bd = d; best = i; } });
  return best;
}
$("scatter").onmousemove = e => {
  const i = nearest(e, $("scatter"), pts), tip = $("tip");
  if (i !== hover) { hover = i; drawScatter(); }
  if (i < 0) { tip.style.display = "none"; return; }
  const s = S[i], r = $("scatter").getBoundingClientRect();
  tip.innerHTML = `<img src="${s.img}"><div class="${s.label ? "c-right" : "c-left"}">${labelName(s.label)}</div><div class="mono">${s.name} · ${t(s.split)}</div>`;
  tip.style.display = "block";
  const tx = e.clientX - r.left + 16; tip.style.left = (tx + 150 > r.width ? tx - 180 : tx) + "px"; tip.style.top = Math.max(0, e.clientY - r.top - 70) + "px";
};
$("scatter").onmouseleave = () => { hover = -1; $("tip").style.display = "none"; drawScatter(); };
$("scatter").onclick = e => { const i = nearest(e, $("scatter"), pts); if (i >= 0) { selected = i; drawScatter(); sampleCard(); } };
function probBlock(name, p, label) {
  const pred = p > .5 ? 1 : 0, ok = pred === label;
  return `<div class="prob"><b>${name}</b> · ${labelName(pred)} <span dir="ltr">${pct(Math.max(p, 1 - p))}</span><span class="badge ${ok ? "ok" : "no"}">${t(ok ? "ok" : "no")}</span>
    <div class="prob-bar"><div style="width:${p * 100}%"></div></div></div>`;
}
function sampleCard() {
  if (selected < 0) { $("sampleCard").innerHTML = `<p class="note">${t("sel.empty")}</p>`; return; }
  const s = S[selected], test = s.split === "test";
  $("sampleCard").innerHTML = `<img src="${s.img}"><div class="mono">${s.name} · ${t("sel.true")}: <span class="${s.label ? "c-right" : "c-left"}">${labelName(s.label)}</span></div>
    <p class="note">${t(test ? "sel.test" : "sel.train")}</p>${probBlock("SVM", test ? s.svmProb : s.cvSvm, s.label)}${probBlock("Random Forest", test ? s.rfProb : s.cvRf, s.label)}
    <button class="btn small" id="openInCv">↑ ${t("sel.open")}</button>`;
  $("openInCv").onclick = () => { selectSample(selected); $("vision").scrollIntoView(); };
}
rerenders.push(drawScatter, sampleCard);

/* ------------------------------------------------------------ variance / confusion / compare */
function drawVariance() {
  const c = $("variance"), ctx = c.getContext("2d"), W = c.width, H = c.height, pad = { l: 34, r: 8, t: 12, b: 24 }, ev = D.explainedVariance;
  ctx.clearRect(0, 0, W, H);
  const bw = (W - pad.l - pad.r) / ev.length, sy = v => H - pad.b - v * (H - pad.t - pad.b);
  ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.font = "10px JetBrains Mono, monospace"; ctx.textAlign = "right";
  for (let k = 0; k <= 4; k++) { const y = sy(k / 4); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(k * 25 + "%", pad.l - 4, y + 3); }
  let cum = 0; const cp = [];
  ev.forEach((v, k) => {
    cum += v; ctx.fillStyle = "#b79cff"; ctx.beginPath(); ctx.roundRect(pad.l + k * bw + 2, sy(v), bw - 4, sy(0) - sy(v), [3, 3, 0, 0]); ctx.fill();
    cp.push([pad.l + k * bw + bw / 2, sy(cum)]);
    if (k % 4 === 0) { ctx.fillStyle = MUTED; ctx.textAlign = "center"; ctx.fillText(k + 1, pad.l + k * bw + bw / 2, H - pad.b + 13); }
  });
  ctx.strokeStyle = SIGNAL; ctx.lineWidth = 2; ctx.beginPath(); cp.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
  ctx.fillStyle = SIGNAL; ctx.textAlign = "right"; ctx.fillText(t("var.cum"), W - pad.r, cp[cp.length - 1][1] - 8);
  $("varNote").innerHTML = t("var.n", { a: pct(ev[0]), k: ev.length, b: pct(cum) });
}
let cmModel = "svm";
function renderCM(el, cm) {
  const tot = cm.flat().reduce((a, b) => a + b, 0), mx = Math.max(...cm.flat());
  const cell = (v, ok) => `<div class="cell" style="background:rgba(${ok ? "110,240,165" : "255,111,97"},${.08 + .5 * v / mx})">${v}</div>`;
  el.innerHTML = `<div></div><div class="hd">${t("cm.pl")}</div><div class="hd">${t("cm.pr")}</div>
    <div class="hd">${t("cm.tl")}</div>${cell(cm[0][0], 1)}${cell(cm[0][1], 0)}<div class="hd">${t("cm.tr")}</div>${cell(cm[1][0], 0)}${cell(cm[1][1], 1)}
    <div></div><div class="hd" style="grid-column:span 2">${t("cm.acc")} <span dir="ltr">${pct((cm[0][0] + cm[1][1]) / tot)}</span></div>`;
}
document.querySelectorAll("#cmModel button").forEach(b => b.onclick = () => { cmModel = b.dataset.m; document.querySelectorAll("#cmModel button").forEach(o => o.classList.toggle("on", o === b)); renderCM($("cm"), D[cmModel].cm); });
function drawCompare() {
  const c = $("compare"), ctx = c.getContext("2d"), W = c.width, H = c.height, pad = { l: 34, r: 8, t: 18, b: 44 }, sy = v => H - pad.b - (v / 100) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.font = "10px JetBrains Mono, monospace";
  ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.textAlign = "right";
  for (let k = 0; k <= 4; k++) { const y = sy(k * 25); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(k * 25 + "%", pad.l - 4, y + 3); }
  const gw = (W - pad.l - pad.r) / 2, bw = gw / 3.2;
  [["SVM", D.svm], ["Random Forest", D.rf]].forEach(([name, m], g) => {
    [[m.acc, "rgba(236,233,226,.35)"], [m.cvAcc, "#b79cff"]].forEach(([v, col], k) => {
      const x = pad.l + g * gw + gw / 2 - bw + k * bw + 3;
      ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(x, sy(v), bw - 6, sy(0) - sy(v), [4, 4, 0, 0]); ctx.fill();
      ctx.fillStyle = "#ece9e2"; ctx.textAlign = "center"; ctx.font = "500 11px JetBrains Mono, monospace"; ctx.fillText(v + "%", x + (bw - 6) / 2, sy(v) - 5);
    });
    ctx.fillStyle = MUTED; ctx.font = "11px Inter, sans-serif"; ctx.fillText(name, pad.l + g * gw + gw / 2, H - pad.b + 16);
  });
  ctx.setLineDash([4, 4]); ctx.strokeStyle = "#ffb547"; ctx.beginPath(); ctx.moveTo(pad.l, sy(50)); ctx.lineTo(W - pad.r, sy(50)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#ffb547"; ctx.textAlign = "right"; ctx.font = "10px JetBrains Mono, monospace"; ctx.fillText(`${t("cmp.chance")} 50%`, W - pad.r, sy(50) - 5);
  ctx.textAlign = "left"; ctx.fillStyle = "rgba(236,233,226,.35)"; ctx.fillRect(pad.l, H - 13, 9, 9); ctx.fillStyle = MUTED; ctx.fillText(t("cmp.split"), pad.l + 14, H - 5);
  ctx.fillStyle = "#b79cff"; ctx.fillRect(pad.l + 140, H - 13, 9, 9); ctx.fillStyle = MUTED; ctx.fillText(t("cmp.cv"), pad.l + 154, H - 5);
}
rerenders.push(drawVariance, () => renderCM($("cm"), D[cmModel].cm), drawCompare);

/* ------------------------------------------------------------ k-NN lab */
let seed = 42;
function runKnn() {
  const k = +$("kVal").value, d = +$("nPc").value, { train, test } = split(seed, +$("testPct").value), cm = [[0, 0], [0, 0]], pred = new Map();
  test.forEach(i => { const p = knn(S[i].pca, train, k, d); pred.set(i, p); cm[S[i].label][p]++; });
  const acc = (cm[0][0] + cm[1][1]) / test.length;
  $("knnAcc").textContent = pct(acc); $("knnAcc").style.color = acc >= .6 ? OK : acc >= .5 ? SIGNAL : BAD;
  renderCM($("knnCm"), cm);
  const c = $("knnMap"), ctx = c.getContext("2d"), W = c.width, H = c.height, pad = { l: 44, r: 12, t: 12, b: 34 };
  const xr = range(S.map(s => s.pca[0])), yr = range(S.map(s => s.pca[1]));
  ctx.fillStyle = "#0b0d13"; ctx.fillRect(0, 0, W, H);
  if (d <= 2) decisionMap(ctx, W, H, pad, xr, yr, train, k, d);
  axes(ctx, W, H, pad, xr, yr, "PC1", "PC2");
  const sx = v => pad.l + ((v - xr[0]) / (xr[1] - xr[0])) * (W - pad.l - pad.r), sy = v => H - pad.b - ((v - yr[0]) / (yr[1] - yr[0])) * (H - pad.t - pad.b);
  train.forEach(i => { ctx.fillStyle = (S[i].label ? RIGHT : LEFT) + "88"; ctx.strokeStyle = "#0b0d13"; ctx.lineWidth = 1; marker(ctx, sx(S[i].pca[0]), sy(S[i].pca[1]), 4, false); });
  test.forEach(i => { ctx.fillStyle = S[i].label ? RIGHT : LEFT; ctx.strokeStyle = pred.get(i) === S[i].label ? OK : "#ff2d2d"; ctx.lineWidth = 2; marker(ctx, sx(S[i].pca[0]), sy(S[i].pca[1]), 5, true); });
  $("knnNote").textContent = d <= 2 ? t("knn.n2") : t("knn.nd", { d });
}
for (const id of ["kVal", "nPc", "testPct"]) $(id).oninput = () => { $(id + "V").textContent = $(id).value; runKnn(); };
$("reshuffle").onclick = () => { seed = Math.floor(Math.random() * 1e9); runKnn(); };
rerenders.push(runKnn);

/* ------------------------------------------------------------ game */
let gIdx = 0, you = [0, 0], svm = [0, 0], answer = null;
async function gameNext() {
  gIdx = Math.floor(Math.random() * S.length); answer = null; gameText();
  const p = toPixels(await loadImg(S[gIdx]));
  putRGB($("gameImg"), $("gameEnhance").checked ? activityRGB(recoverActivity(p, circleMask(64)), true) : i => [p[i * 4], p[i * 4 + 1], p[i * 4 + 2]]);
}
function gameText() {
  if (answer === null) { $("gameResult").textContent = t("play.q"); return; }
  const s = S[gIdx], sp = s.cvSvmPred, b = ok => `<span class="badge ${ok ? "ok" : "no"}">${ok ? "✓" : "✗"}</span>`;
  $("gameResult").innerHTML = `${t("play.ans")}: <span class="${s.label ? "c-right" : "c-left"}">${labelName(s.label)}</span><br>
    <small class="note">${t("play.you")}: ${labelName(answer)} ${b(answer === s.label)} · SVM: ${labelName(sp)} ${b(sp === s.label)}</small>`;
}
function guess(g) {
  if (answer !== null) return; answer = g;
  const s = S[gIdx], sp = s.cvSvmPred;
  you[1]++; svm[1]++; if (g === s.label) you[0]++; if (sp === s.label) svm[0]++;
  $("sYou").textContent = `${you[0]} / ${you[1]}`; $("sSvm").textContent = `${svm[0]} / ${svm[1]}`; gameText();
}
$("gL").onclick = () => guess(0); $("gR").onclick = () => guess(1); $("gNext").onclick = gameNext; $("gameEnhance").onchange = gameNext;
rerenders.push(gameText, () => { $("lim3").innerHTML = t("lim.3", { b: D.svm.cvAcc, c: D.rf.cvAcc }); });

/* ------------------------------------------------------------ reveal + nav */
const revealObs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); revealObs.unobserve(e.target); } }), { threshold: .12 });
document.querySelectorAll(".chapter-head, .tile, .panel, .limit, .game, .picker").forEach(el => { el.classList.add("reveal"); revealObs.observe(el); });
const navObs = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) document.querySelectorAll(".nav-links a").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
}), { rootMargin: "-50% 0px -50% 0px" });
document.querySelectorAll(".chapter").forEach(s => navObs.observe(s));

/* ------------------------------------------------------------ init */
applyLang();
selectSample(storyIdx);
gameNext();
})();

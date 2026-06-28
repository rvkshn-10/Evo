const API_BASE = import.meta.env.VITE_API_BASE || "";

let evoData = null;
let animationFrame = null;
let modalOpen = false;

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok || !contentType.includes("application/json")) return null;
  return response.json();
}

const PALETTE = {
  input: { fill: "#0ea5e9", stroke: "#7dd3fc", glow: "rgba(14,165,233,0.35)" },
  dense: { fill: "#8b5cf6", stroke: "#c4b5fd", glow: "rgba(139,92,246,0.3)" },
  tree: { fill: "#10b981", stroke: "#6ee7b7", glow: "rgba(16,185,129,0.3)" },
  branch: { fill: "#6366f1", stroke: "#a5b4fc", glow: "rgba(99,102,241,0.3)" },
  output: { fill: "#f59e0b", stroke: "#fcd34d", glow: "rgba(245,158,11,0.35)" },
  panel: "rgba(15,23,42,0.55)",
  panelBorder: "rgba(148,163,184,0.18)",
  synapse: "rgba(56,189,248,0.14)",
  synapseHot: "rgba(167,139,250,0.45)",
};

function layerStyle(layer) {
  const type = layer.type || "dense";
  if (type === "input") return PALETTE.input;
  if (type === "output") return PALETTE.output;
  if (type === "tree") return PALETTE.tree;
  if (type === "branch") return PALETTE.branch;
  return PALETTE.dense;
}

function neuronCountForLayer(layer) {
  const label = layer.label || layer.id || "";
  const match = label.match(/(\d+)/);
  if (match) {
    const n = parseInt(match[1], 10);
    return Math.min(8, Math.max(4, Math.round(n / 16)));
  }
  if (layer.type === "input") return 14;
  if (layer.type === "output") return 1;
  if (layer.type === "tree") return 5;
  return 6;
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(rect.width || canvas.width, 320);
  const h = Math.max(rect.height || canvas.height, 280);
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function buildLayout(architecture, w, h) {
  const layers = architecture.layers || [];
  const edges = architecture.edges || [];
  const padX = 56;
  const padY = 44;
  const maxCol = Math.max(...layers.map((l) => (Number.isFinite(l.column) ? l.column : 0)), 1);

  const columns = [];
  layers.forEach((layer, index) => {
    const column = Number.isFinite(layer.column) ? layer.column : index;
    const lane = Number.isFinite(layer.lane) ? layer.lane : 0.5;
    const count = neuronCountForLayer(layer);
    const x = padX + (column / maxCol) * (w - padX * 2);
    const neurons = Array.from({ length: count }, (_, i) => ({ index: i }));
    const spacing = Math.min(20, (h - padY * 2) / (count + 1));
    const totalH = (count - 1) * spacing;
    const cy = padY + lane * (h - padY * 2);
    const startY = cy - totalH / 2;

    const positions = neurons.map((n, i) => ({
      x,
      y: startY + i * spacing,
      layer,
      style: layerStyle(layer),
      ni: i,
    }));

    const label = (layer.label || layer.id).replace(/\\n/g, "\n");
  const panelTop = Math.min(...positions.map((p) => p.y)) - 22;
  const panelBottom = Math.max(...positions.map((p) => p.y)) + 22;

    columns.push({
      layer,
      column,
      lane,
      x,
      positions,
      panel: { top: panelTop, bottom: panelBottom, left: x - 34, right: x + 34 },
      label,
      type: layer.type || "dense",
    });
  });

  columns.sort((a, b) => a.column - b.column || a.lane - b.lane);

  const byLayerId = new Map(columns.map((c) => [c.layer.id, c.positions]));

  return { columns, edges, byLayerId, padX, w, h };
}

function drawBackground(ctx, w, h) {
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, "#070b14");
  bg.addColorStop(0.45, "#0c1220");
  bg.addColorStop(1, "#060910");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const vignette = ctx.createRadialGradient(w / 2, h / 2, h * 0.2, w / 2, h / 2, h * 0.85);
  vignette.addColorStop(0, "rgba(56,189,248,0.04)");
  vignette.addColorStop(1, "rgba(0,0,0,0.45)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(255,255,255,0.025)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 48) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 48) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}

function drawLayerPanels(ctx, columns) {
  columns.forEach((col) => {
    const { panel, type } = col;
    const height = panel.bottom - panel.top;
    ctx.fillStyle = PALETTE.panel;
    ctx.strokeStyle = PALETTE.panelBorder;
    ctx.lineWidth = 1;
    roundRect(ctx, panel.left, panel.top, panel.right - panel.left, height, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "rgba(226,232,240,0.9)";
    ctx.font = "600 11px 'DM Sans', system-ui, sans-serif";
    ctx.textAlign = "center";
    const lines = col.label.split("\n");
    lines.forEach((line, i) => {
      ctx.fillText(line, col.x, panel.top - 14 - (lines.length - 1 - i) * 13);
    });

    ctx.fillStyle = "rgba(148,163,184,0.75)";
    ctx.font = "500 9px 'DM Sans', system-ui, sans-serif";
    ctx.fillText(type.toUpperCase(), col.x, panel.bottom + 14);
  });
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawSynapses(ctx, edges, byLayerId, time) {
  edges.forEach(([fromId, toId], edgeIndex) => {
    const fromList = byLayerId.get(fromId) || [];
    const toList = byLayerId.get(toId) || [];
    if (!fromList.length || !toList.length) return;

    const pairs = Math.min(fromList.length, toList.length, 10);
    for (let i = 0; i < pairs; i++) {
      const a = fromList[Math.floor((i / pairs) * fromList.length)];
      const b = toList[Math.floor((i / pairs) * toList.length)];
      const weight = 0.25 + ((edgeIndex * 5 + i * 11) % 9) / 12;
      const midX = (a.x + b.x) / 2;
      const midY = (a.y + b.y) / 2 - 12;

      ctx.strokeStyle = `rgba(56,189,248,${0.06 + weight * 0.12})`;
      ctx.lineWidth = 0.6 + weight * 0.8;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(midX, midY, b.x, b.y);
      ctx.stroke();

      const phase = (time * 0.35 + edgeIndex * 0.1 + i * 0.08) % 1;
      const t = phase;
      const px = (1 - t) ** 2 * a.x + 2 * (1 - t) * t * midX + t ** 2 * b.x;
      const py = (1 - t) ** 2 * a.y + 2 * (1 - t) * t * midY + t ** 2 * b.y;
      ctx.fillStyle = PALETTE.synapseHot;
      ctx.globalAlpha = 0.25 + weight * 0.35;
      ctx.beginPath();
      ctx.arc(px, py, 1.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  });
}

function drawNode(ctx, pos, time) {
  const { x, y, layer, style, ni } = pos;
  const type = layer.type || "dense";
  const pulse = 0.5 + 0.5 * Math.sin(time * 1.2 + ni * 0.35);
  const r = type === "output" ? 9 : type === "tree" ? 6.5 : 5.5;

  ctx.save();
  ctx.shadowColor = style.glow;
  ctx.shadowBlur = 8 + pulse * 4;

  if (type === "tree") {
    ctx.fillStyle = style.fill;
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r, y);
    ctx.lineTo(x, y + r);
    ctx.lineTo(x - r, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  } else if (type === "input") {
    const grad = ctx.createRadialGradient(x - 1, y - 1, 0, x, y, r + 2);
    grad.addColorStop(0, "#e0f2fe");
    grad.addColorStop(0.35, style.fill);
    grad.addColorStop(1, "#0369a1");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  } else {
    const grad = ctx.createRadialGradient(x - 1.5, y - 1.5, 0, x, y, r + 1);
    grad.addColorStop(0, "#f8fafc");
    grad.addColorStop(0.3, style.fill);
    grad.addColorStop(1, "rgba(15,23,42,0.9)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.restore();
}

function drawLegend(ctx, w, h, architecture) {
  const items = [
    { label: "Input features", color: PALETTE.input.fill },
    { label: "MLP head", color: PALETTE.dense.fill },
    { label: "Tree head", color: PALETTE.tree.fill },
    { label: "Outputs", color: PALETTE.output.fill },
  ];
  const hasTree = (architecture.layers || []).some((l) => l.type === "tree");
  const visible = hasTree ? items : items.filter((i) => i.label !== "Tree head");

  ctx.font = "500 10px 'DM Sans', system-ui, sans-serif";
  ctx.textAlign = "left";
  let x = 16;
  const y = h - 12;
  visible.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(x + 4, y - 3, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(148,163,184,0.85)";
    ctx.fillText(item.label, x + 12, y);
    x += ctx.measureText(item.label).width + 28;
  });
}

function drawNetworkGraph(canvas, architecture, time = 0) {
  const { ctx, w, h } = setupCanvas(canvas);
  drawBackground(ctx, w, h);

  if (!architecture?.layers?.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px 'DM Sans', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Architecture data unavailable", w / 2, h / 2);
    return;
  }

  const layout = buildLayout(architecture, w, h);
  drawLayerPanels(ctx, layout.columns);
  drawSynapses(ctx, layout.edges, layout.byLayerId, time);
  layout.columns.forEach((col) => col.positions.forEach((pos) => drawNode(ctx, pos, time)));
  drawLegend(ctx, w, h, architecture);

  const title = architecture.name || architecture.selected_model || "Evo dual-head";
  ctx.fillStyle = "rgba(226,232,240,0.92)";
  ctx.font = "600 12px 'DM Sans', system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(title, 16, 20);
  ctx.fillStyle = "rgba(148,163,184,0.7)";
  ctx.font = "500 10px 'DM Sans', system-ui, sans-serif";
  ctx.fillText("hazard + occupancy → success % · evacuation time", 16, 34);
}

function drawLossChart(canvas, metrics) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const train = metrics.train_loss || [];
  const val = metrics.val_loss || [];
  if (!train.length && !val.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px 'DM Sans', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(metrics.message || "Training metrics pending", w / 2, h / 2);
    return;
  }

  const maxY = Math.max(...train, ...(val.length ? val : [0])) * 1.08 || 1;
  const pad = { t: 28, r: 16, b: 28, l: 36 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  ctx.fillStyle = "rgba(0,0,0,0.2)";
  roundRect(ctx, pad.l, pad.t, plotW, plotH, 8);
  ctx.fill();

  const plot = (data, color, fill) => {
    if (!data.length) return;
    ctx.beginPath();
    data.forEach((y, i) => {
      const x = pad.l + (i / Math.max(data.length - 1, 1)) * plotW;
      const yy = pad.t + plotH - (y / maxY) * plotH;
      if (i === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    });
    ctx.lineTo(pad.l + plotW, pad.t + plotH);
    ctx.lineTo(pad.l, pad.t + plotH);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((y, i) => {
      const x = pad.l + (i / Math.max(data.length - 1, 1)) * plotW;
      const yy = pad.t + plotH - (y / maxY) * plotH;
      if (i === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    });
    ctx.stroke();
  };

  plot(train, "#38bdf8", "rgba(56,189,248,0.1)");
  plot(val, "#fbbf24", "rgba(251,191,36,0.08)");

  ctx.font = "500 10px 'DM Sans', system-ui, sans-serif";
  ctx.fillStyle = "#38bdf8";
  ctx.textAlign = "left";
  ctx.fillText("Train", pad.l, 16);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText("Validation", pad.l + 44, 16);
}

function renderMetrics(data) {
  const metrics = data.metrics || {};
  const format = (value, digits = 3) =>
    Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
  const gateStatus = metrics.all_quality_gates_pass;
  const approved = gateStatus === true;
  const backend = data.backend || "—";
  const runtimeLabel =
    backend === "openvino" ? "OpenVINO (CPU)" : backend === "onnxruntime" ? "ONNX Runtime (CPU)" : backend;

  document.getElementById("evoMetrics").innerHTML = `
    <span class="status-badge">${approved ? "Production approved" : "Research preview"}</span>
    <div class="metric-row"><span class="metric-label">Val MAE success</span><span class="metric-value">${format(metrics.val_mae_success_pct)}%</span></div>
    <div class="metric-row"><span class="metric-label">Val R² success</span><span class="metric-value">${format(metrics.val_r2_success_pct)}</span></div>
    <div class="metric-row"><span class="metric-label">Val MAE time</span><span class="metric-value">${format(metrics.val_mae_time_min)} min</span></div>
    <div class="metric-row"><span class="metric-label">Val R² time</span><span class="metric-value">${format(metrics.val_r2_time_min)}</span></div>
    <div class="metric-row"><span class="metric-label">Quality gates</span><span class="metric-value">${gateStatus == null ? "—" : gateStatus ? "Passed" : "Failed"}</span></div>
    <div class="metric-row"><span class="metric-label">Runtime</span><span class="metric-value">${runtimeLabel}</span></div>
  `;
}

function renderEvoViz() {
  if (!evoData) return;
  const netCanvas = document.getElementById("evoNetworkCanvas");
  const lossCanvas = document.getElementById("evoLossCanvas");
  if (lossCanvas) drawLossChart(lossCanvas, evoData.metrics || {});
  renderMetrics(evoData);
  if (netCanvas) drawNetworkGraph(netCanvas, evoData.architecture || {}, 0);
}

function startNetworkAnimation() {
  const netCanvas = document.getElementById("evoNetworkCanvas");
  if (!netCanvas || !evoData?.architecture) return;

  let start = null;
  const tick = (ts) => {
    if (!modalOpen) {
      animationFrame = null;
      return;
    }
    if (!start) start = ts;
    drawNetworkGraph(netCanvas, evoData.architecture, (ts - start) / 1000);
    animationFrame = requestAnimationFrame(tick);
  };
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = requestAnimationFrame(tick);
}

function stopNetworkAnimation() {
  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }
}

function openEvoModal() {
  const modal = document.getElementById("evoModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  modalOpen = true;
  document.body.style.overflow = "hidden";
  renderEvoViz();
  startNetworkAnimation();
}

function closeEvoModal() {
  const modal = document.getElementById("evoModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modalOpen = false;
  document.body.style.overflow = "";
  stopNetworkAnimation();
}

async function loadEvoData() {
  const data = await fetchJson("/api/evo/visualization");
  if (!data) {
    document.getElementById("evoStatus").textContent = "Model info unavailable — start the API";
    return;
  }

  evoData = data;
  const modelVersion = data.model_version || "evo1.2";
  const backend =
    data.backend === "openvino"
      ? "OpenVINO"
      : data.backend === "onnxruntime"
        ? "ONNX Runtime"
        : data.backend || "ready";
  document.getElementById("evoStatus").textContent = data.available
    ? `${modelVersion} · ${backend}`
    : `${modelVersion} · awaiting model artifacts`;
}

export async function initEvoVisualization() {
  const showBtn = document.getElementById("showEvoBtn");
  const closeBtn = document.getElementById("evoModalClose");
  const backdrop = document.getElementById("evoModalBackdrop");

  showBtn?.addEventListener("click", openEvoModal);
  closeBtn?.addEventListener("click", closeEvoModal);
  backdrop?.addEventListener("click", closeEvoModal);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modalOpen) closeEvoModal();
  });

  window.addEventListener("resize", () => {
    if (modalOpen) renderEvoViz();
  });

  await loadEvoData();
}

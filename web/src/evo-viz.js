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

function neuronCountForLayer(layer) {
  const label = layer.label || layer.id || "";
  const match = label.match(/(\d+)/);
  if (match) {
    const n = parseInt(match[1], 10);
    return Math.min(10, Math.max(4, Math.round(n / 12)));
  }
  if (layer.type === "input") return 12;
  if (layer.type === "output") return 2;
  if (layer.type === "branch") return 6;
  return 7;
}

function layerColor(layer) {
  if (layer.type === "input") return { core: "#38bdf8", glow: "rgba(56,189,248,0.45)" };
  if (layer.type === "output") return { core: "#fbbf24", glow: "rgba(251,191,36,0.4)" };
  if (layer.type === "branch") return { core: "#a78bfa", glow: "rgba(167,139,250,0.35)" };
  return { core: "#f472b6", glow: "rgba(244,114,182,0.35)" };
}

function buildLayerColumns(architecture) {
  const layers = architecture.layers || [];
  const edges = architecture.edges || [];
  const hasLayout = layers.some(
    (layer) => Number.isFinite(layer.column) || Number.isFinite(layer.lane),
  );

  const columns = [];
  const layerMap = new Map();

  layers.forEach((layer, index) => {
    const column = Number.isFinite(layer.column) ? layer.column : index;
    const lane = Number.isFinite(layer.lane) ? layer.lane : 0.5;
    const count = neuronCountForLayer(layer);
    const neurons = Array.from({ length: count }, (_, i) => ({
      id: `${layer.id}_n${i}`,
      layerId: layer.id,
      index: i,
    }));

    const entry = {
      layer,
      column,
      lane,
      neurons,
      color: layerColor(layer),
    };
    layerMap.set(layer.id, entry);

    let col = columns.find((c) => c.column === column && c.lane === lane);
    if (!col) {
      col = { column, lane, stacks: [] };
      columns.push(col);
    }
    col.stacks.push(entry);
  });

  columns.sort((a, b) => a.column - b.column || a.lane - b.lane);

  return { columns, edges, hasLayout, layerMap };
}

function drawNeuron(ctx, x, y, radius, color, pulse = 0) {
  const glowR = radius + 6 + pulse * 4;
  const grad = ctx.createRadialGradient(x, y, 0, x, y, glowR);
  grad.addColorStop(0, color.core);
  grad.addColorStop(0.55, color.core);
  grad.addColorStop(1, "rgba(0,0,0,0)");

  ctx.save();
  ctx.globalAlpha = 0.35 + pulse * 0.15;
  ctx.fillStyle = color.glow;
  ctx.beginPath();
  ctx.arc(x, y, glowR, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  const body = ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.35, 0, x, y, radius);
  body.addColorStop(0, "#ffffff");
  body.addColorStop(0.25, color.core);
  body.addColorStop(1, "rgba(0,0,0,0.55)");

  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = 0.8;
  ctx.stroke();
}

function drawSynapse(ctx, ax, ay, bx, by, weight, t) {
  const midX = (ax + bx) / 2;
  const midY = (ay + by) / 2 - 18 - weight * 8;
  const alpha = 0.08 + weight * 0.22;

  ctx.strokeStyle = `rgba(56,189,248,${alpha})`;
  ctx.lineWidth = 0.5 + weight * 1.2;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.quadraticCurveTo(midX, midY, bx, by);
  ctx.stroke();

  const dotT = (t * 0.4 + weight) % 1;
  const px = (1 - dotT) * (1 - dotT) * ax + 2 * (1 - dotT) * dotT * midX + dotT * dotT * bx;
  const py = (1 - dotT) * (1 - dotT) * ay + 2 * (1 - dotT) * dotT * midY + dotT * dotT * by;

  ctx.fillStyle = `rgba(167,139,250,${0.35 + weight * 0.3})`;
  ctx.beginPath();
  ctx.arc(px, py, 1.2 + weight, 0, Math.PI * 2);
  ctx.fill();
}

function drawNetworkGraph(canvas, architecture, time = 0) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, "rgba(8,12,24,0.95)");
  bg.addColorStop(0.5, "rgba(12,18,36,0.98)");
  bg.addColorStop(1, "rgba(6,10,20,0.95)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(255,255,255,0.03)";
  ctx.lineWidth = 1;
  for (let gx = 40; gx < w; gx += 40) {
    ctx.beginPath();
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, h);
    ctx.stroke();
  }
  for (let gy = 40; gy < h; gy += 40) {
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(w, gy);
    ctx.stroke();
  }

  const { columns, edges } = buildLayerColumns(architecture);
  if (!columns.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px DM Sans, system-ui";
    ctx.textAlign = "center";
    ctx.fillText("Architecture data unavailable", w / 2, h / 2);
    return;
  }

  const padX = 70;
  const padY = 50;
  const maxCol = Math.max(...columns.map((c) => c.column), 1);
  const positions = {};

  columns.forEach((col) => {
    const x = padX + (col.column / maxCol) * (w - padX * 2);
    const stackCount = col.stacks.length;
    col.stacks.forEach((stack, si) => {
      const laneOffset = stackCount > 1 ? (si - (stackCount - 1) / 2) * 0.22 : 0;
      const baseY = padY + (col.lane + laneOffset) * (h - padY * 2);
      const spacing = Math.min(22, (h - padY * 2) / (stack.neurons.length + 2));
      const totalH = (stack.neurons.length - 1) * spacing;
      const startY = baseY - totalH / 2;

      stack.neurons.forEach((neuron, ni) => {
        positions[neuron.id] = {
          x,
          y: startY + ni * spacing,
          color: stack.color,
          layer: stack.layer,
          ni,
          count: stack.neurons.length,
        };
      });

      stack.labelY = startY - 28;
      stack.labelX = x;
    });
  });

  const layerNeurons = new Map();
  Object.entries(positions).forEach(([id, pos]) => {
    const lid = pos.layer.id;
    if (!layerNeurons.has(lid)) layerNeurons.set(lid, []);
    layerNeurons.get(lid).push(pos);
  });

  edges.forEach(([fromId, toId], edgeIndex) => {
    const fromList = layerNeurons.get(fromId) || [];
    const toList = layerNeurons.get(toId) || [];
    if (!fromList.length || !toList.length) return;

    const pairs = Math.min(fromList.length, toList.length, 14);
    for (let i = 0; i < pairs; i++) {
      const fi = Math.floor((i / pairs) * fromList.length);
      const ti = Math.floor((i / pairs) * toList.length);
      const a = fromList[fi];
      const b = toList[ti];
      const weight = 0.3 + ((edgeIndex * 7 + i * 13) % 10) / 14;
      drawSynapse(ctx, a.x, a.y, b.x, b.y, weight, time + i * 0.07);
    }
  });

  Object.values(positions).forEach((pos) => {
    const pulse = 0.5 + 0.5 * Math.sin(time * 2 + pos.ni * 0.4);
    drawNeuron(ctx, pos.x, pos.y, 7, pos.color, pulse * 0.12);
  });

  columns.forEach((col) => {
    col.stacks.forEach((stack) => {
      const label = (stack.layer.label || stack.layer.id).replace(/\\n/g, "\n");
      const lines = label.split("\n");
      ctx.fillStyle = "rgba(238,242,255,0.85)";
      ctx.font = "600 11px DM Sans, system-ui";
      ctx.textAlign = "center";
      lines.forEach((line, i) => {
        ctx.fillText(line, stack.labelX, stack.labelY + i * 14);
      });

      ctx.fillStyle = "rgba(148,163,184,0.7)";
      ctx.font = "10px DM Sans, system-ui";
      ctx.fillText(stack.layer.type || "dense", stack.labelX, stack.labelY + lines.length * 14 + 4);
    });
  });

  ctx.fillStyle = "rgba(148,163,184,0.5)";
  ctx.font="10px DM Sans, system-ui";
  ctx.textAlign = "left";
  ctx.fillText("← hazard features", padX - 10, h - 16);
  ctx.textAlign = "right";
  ctx.fillText("evacuation outputs →", w - padX + 10, h - 16);
}

function drawLossChart(canvas, metrics) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const train = metrics.train_loss || [];
  const val = metrics.val_loss || [];
  if (!train.length && !val.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px DM Sans, system-ui";
    ctx.textAlign = "center";
    ctx.fillText(metrics.message || "Training metrics pending", w / 2, h / 2);
    return;
  }

  const series = train.length ? train : val;
  const maxY = Math.max(...series, ...(val.length ? val : [0])) * 1.1 || 1;
  const pad = 28;

  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.fillRect(pad, pad, w - pad * 2, h - pad * 2);

  const plot = (data, color, fillAlpha) => {
    if (!data.length) return;
    ctx.beginPath();
    data.forEach((y, i) => {
      const x = pad + (i / Math.max(data.length - 1, 1)) * (w - pad * 2);
      const yy = h - pad - (y / maxY) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    });
    ctx.lineTo(pad + (w - pad * 2), h - pad);
    ctx.lineTo(pad, h - pad);
    ctx.closePath();
    ctx.fillStyle = fillAlpha;
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((y, i) => {
      const x = pad + (i / Math.max(data.length - 1, 1)) * (w - pad * 2);
      const yy = h - pad - (y / maxY) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    });
    ctx.stroke();
  };

  plot(train, "#38bdf8", "rgba(56,189,248,0.12)");
  plot(val, "#fbbf24", "rgba(251,191,36,0.08)");

  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px DM Sans, system-ui";
  ctx.textAlign = "left";
  ctx.fillText("● Train", pad, 16);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText("● Val", pad + 58, 16);
}

function renderMetrics(data) {
  const metrics = data.metrics || {};
  const format = (value, digits = 3) =>
    Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
  const gateStatus = metrics.all_quality_gates_pass;
  const approved = gateStatus === true;

  document.getElementById("evoMetrics").innerHTML = `
    <span class="status-badge">${approved ? "Production approved" : "Research preview"}</span>
    <div class="metric-row"><span class="metric-label">Val MAE success</span><span class="metric-value">${format(metrics.val_mae_success_pct)}%</span></div>
    <div class="metric-row"><span class="metric-label">Val R² success</span><span class="metric-value">${format(metrics.val_r2_success_pct)}</span></div>
    <div class="metric-row"><span class="metric-label">Val MAE time</span><span class="metric-value">${format(metrics.val_mae_time_min)} min</span></div>
    <div class="metric-row"><span class="metric-label">Val R² time</span><span class="metric-value">${format(metrics.val_r2_time_min)}</span></div>
    <div class="metric-row"><span class="metric-label">Quality gates</span><span class="metric-value">${gateStatus == null ? "—" : gateStatus ? "Passed" : "Failed"}</span></div>
    <div class="metric-row"><span class="metric-label">Runtime</span><span class="metric-value">${data.backend || "—"}</span></div>
    <div class="metric-row"><span class="metric-label">OpenVINO</span><span class="metric-value">${data.openvino_connected ? "Connected" : "Not loaded"}</span></div>
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
    const t = (ts - start) / 1000;
    drawNetworkGraph(netCanvas, evoData.architecture, t);
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
  document.getElementById("evoStatus").textContent = data.available
    ? `${modelVersion} · ${data.backend || "ready"}`
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

  await loadEvoData();
}

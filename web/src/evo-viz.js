import { glossaryTerm, initGlossaryTooltips } from "./glossary.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const LIVE_POLL_MS = 2800;

let evoData = null;
let liveFlow = null;
let animationFrame = null;
let livePollTimer = null;
let modalOpen = false;

/** Canvas layout + edge hit regions updated each frame. */
let graphState = {
  nodes: new Map(),
  edges: [],
  w: 0,
  h: 0,
};

/** Animated packets traveling along connections. */
let particles = [];
let lastFlowStamp = "";
let lastParticleSpawn = 0;

const THEME = {
  bg0: "#060a10",
  bg1: "#0c121c",
  grid: "rgba(91,141,239,0.04)",
  text: "rgba(226,232,240,0.94)",
  muted: "rgba(148,163,184,0.7)",
  accent: "#5b8def",
  accentGlow: "rgba(91,141,239,0.55)",
  accentDim: "rgba(91,141,239,0.14)",
  success: "#34d399",
  warn: "#fbbf24",
  feed: "#64748b",
  feature: "#475569",
  encoder: "#3b5998",
  model: "#6366f1",
  ensemble: "#8b5cf6",
  output: "#22d3ee",
  line: "rgba(148,163,184,0.18)",
  lineHot: "rgba(91,141,239,0.75)",
  lineData: "rgba(52,211,153,0.45)",
};

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok || !contentType.includes("application/json")) return null;
  return response.json();
}

function isCompactViewport(width) {
  return width < 560;
}

function setupCanvas(canvas, { minWidth = 0, height = 380, fillParent = false } = {}) {
  const scroll = canvas.parentElement;
  const container = scroll?.parentElement || scroll || canvas;
  const containerWidth = container?.getBoundingClientRect().width || minWidth || 320;
  const w = fillParent ? containerWidth : Math.max(containerWidth, minWidth, 320);
  const h = height;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = fillParent ? "100%" : `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h, compact: isCompactViewport(containerWidth) };
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

function nodeType(id) {
  if (["noaa", "usgs", "peoplesense", "gdacs"].includes(id)) return "feed";
  if (id === "encoder") return "encoder";
  if (id === "mlp" || id === "lgbm") return "model";
  if (id === "ensemble") return "ensemble";
  if (id === "success" || id === "time") return "output";
  return "feature";
}

function nodeColor(type, status) {
  if (status === "live" || status === "active") {
    if (type === "feed") return THEME.accent;
    if (type === "output") return THEME.success;
  }
  const map = {
    feed: THEME.feed,
    feature: THEME.feature,
    encoder: THEME.encoder,
    model: THEME.model,
    ensemble: THEME.ensemble,
    output: THEME.output,
  };
  return map[type] || THEME.feature;
}

function buildLiveLayout(flow, w, h, compact) {
  const padX = compact ? 20 : 36;
  const padY = compact ? 36 : 44;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;

  const cols = compact
    ? [0.06, 0.26, 0.44, 0.58, 0.72, 0.88]
    : [0.07, 0.24, 0.4, 0.56, 0.72, 0.9];

  const nodes = new Map();
  const feeds = flow.feeds || [];
  const features = (flow.features || []).slice(0, compact ? 6 : 8);

  feeds.forEach((feed, i) => {
    const y = padY + ((i + 1) / (feeds.length + 1)) * innerH;
    nodes.set(feed.id, {
      id: feed.id,
      x: padX + cols[0] * innerW,
      y,
      label: feed.label.replace(" / ", "\n"),
      short: feed.id.toUpperCase().slice(0, 4),
      type: "feed",
      status: feed.status,
      detail: feed.detail,
    });
  });

  features.forEach((feature, i) => {
    const y = padY + ((i + 1) / (features.length + 1)) * innerH;
    const short = feature.label.split(" ")[0].slice(0, 8);
    nodes.set(feature.key, {
      id: feature.key,
      x: padX + cols[1] * innerW,
      y,
      label: short,
      type: "feature",
      status: "active",
      detail: feature.raw_display,
      value: feature.scaled,
    });
  });

  nodes.set("encoder", {
    id: "encoder",
    x: padX + cols[2] * innerW,
    y: padY + innerH / 2,
    label: "Encoder",
    type: "encoder",
    status: "active",
    detail: `${flow.vector_dim || 34}-dim tensor`,
  });

  nodes.set("mlp", {
    id: "mlp",
    x: padX + cols[3] * innerW,
    y: padY + innerH * 0.35,
    label: "MLP",
    type: "model",
    status: flow.backend ? "active" : "idle",
    detail: "Dual-head neural net",
  });

  nodes.set("lgbm", {
    id: "lgbm",
    x: padX + cols[3] * innerW,
    y: padY + innerH * 0.65,
    label: "LGBM",
    type: "model",
    status: "active",
    detail: "Gradient boosting heads",
  });

  nodes.set("ensemble", {
    id: "ensemble",
    x: padX + cols[4] * innerW,
    y: padY + innerH / 2,
    label: "Ensemble",
    type: "ensemble",
    status: flow.prediction?.inference_mode || "hybrid",
    detail: flow.hybrid_mode ? "k-NN + Evo blend" : "Weighted merge",
  });

  const success = flow.prediction?.success_pct;
  const timeMin = flow.prediction?.time_min;
  nodes.set("success", {
    id: "success",
    x: padX + cols[5] * innerW,
    y: padY + innerH * 0.38,
    label: "Success",
    type: "output",
    status: "live",
    detail: success != null ? `${success}%` : "—",
    value: success != null ? success / 100 : 0,
  });

  nodes.set("time", {
    id: "time",
    x: padX + cols[5] * innerW,
    y: padY + innerH * 0.62,
    label: "Time",
    type: "output",
    status: "live",
    detail: timeMin != null ? `${timeMin} min` : "—",
    value: timeMin != null ? Math.min(1, timeMin / 20) : 0,
  });

  const edges = (flow.edges || [])
    .map((edge) => {
      const from = nodes.get(edge.from);
      const to = nodes.get(edge.to);
      if (!from || !to) return null;
      return {
        ...edge,
        x1: from.x,
        y1: from.y,
        x2: to.x,
        y2: to.y,
      };
    })
    .filter(Boolean);

  return { nodes, edges, w, h, compact };
}

function distToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const cx = x1 + t * dx;
  const cy = y1 + t * dy;
  return Math.hypot(px - cx, py - cy);
}

function findEdgeAt(x, y, threshold = 10) {
  let best = null;
  let bestDist = threshold;
  graphState.edges.forEach((edge, index) => {
    const d = distToSegment(x, y, edge.x1, edge.y1, edge.x2, edge.y2);
    if (d < bestDist) {
      bestDist = d;
      best = { edge, index };
    }
  });
  return best;
}

function spawnInferenceWave(full = true) {
  graphState.edges.forEach((edge, index) => {
    if (!full) {
      const isFeed = ["noaa", "usgs", "peoplesense", "gdacs"].includes(edge.from);
      if (!isFeed) return;
    }
    const count = full && (edge.from === "encoder" || edge.to === "encoder") ? 2 : 1;
    for (let i = 0; i < count; i += 1) {
      particles.push({
        edgeIndex: index,
        t: -i * 0.15,
        speed: 0.35 + Math.random() * 0.25,
        hue: edge.to === "success" || edge.to === "time" ? "success" : "accent",
      });
    }
  });
}

function drawHexGrid(ctx, w, h, time) {
  const size = 22;
  ctx.strokeStyle = THEME.grid;
  ctx.lineWidth = 0.5;
  for (let y = -size; y < h + size; y += size * 1.5) {
    for (let x = -size; x < w + size; x += size * Math.sqrt(3)) {
      const ox = x + ((Math.floor(y / (size * 1.5)) % 2) * size * Math.sqrt(3)) / 2;
      const pulse = 0.3 + 0.2 * Math.sin(time * 0.8 + ox * 0.02 + y * 0.02);
      ctx.globalAlpha = pulse * 0.35;
      ctx.beginPath();
      for (let i = 0; i < 6; i += 1) {
        const angle = (Math.PI / 3) * i - Math.PI / 6;
        const hx = ox + size * 0.45 * Math.cos(angle);
        const hy = y + size * 0.45 * Math.sin(angle);
        if (i === 0) ctx.moveTo(hx, hy);
        else ctx.lineTo(hx, hy);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

function drawBackground(ctx, w, h, time) {
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, THEME.bg0);
  bg.addColorStop(0.5, "#0a1018");
  bg.addColorStop(1, THEME.bg1);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  drawHexGrid(ctx, w, h, time);
}

function drawEdge(ctx, edge, time, hovered) {
  const { x1, y1, x2, y2, value = 0.3 } = edge;
  const activity = 0.4 + 0.6 * Math.min(1, Math.abs(value));
  const breathe = 0.7 + 0.3 * Math.sin(time * 2 + x1 * 0.01);

  const grad = ctx.createLinearGradient(x1, y1, x2, y2);
  grad.addColorStop(0, hovered ? THEME.lineHot : `rgba(91,141,239,${0.15 * activity * breathe})`);
  grad.addColorStop(0.5, hovered ? THEME.accentGlow : `rgba(52,211,153,${0.25 * activity})`);
  grad.addColorStop(1, hovered ? THEME.lineHot : `rgba(139,92,246,${0.2 * activity})`);

  ctx.strokeStyle = grad;
  ctx.lineWidth = hovered ? 2.2 : 1 + activity * 0.8;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2 + (y2 - y1) * 0.08;
  ctx.quadraticCurveTo(cx, cy, x2, y2);
  ctx.stroke();

  edge._curve = { cx, cy };
}

function drawParticles(ctx, edges, dt) {
  particles = particles.filter((p) => {
    p.t += p.speed * dt;
    if (p.t > 1.05) return false;
    const edge = edges[p.edgeIndex];
    if (!edge || p.t < 0) return true;

    const t = Math.max(0, Math.min(1, p.t));
    const { x1, y1, x2, y2, _curve } = edge;
    const cx = _curve?.cx ?? (x1 + x2) / 2;
    const cy = _curve?.cy ?? (y1 + y2) / 2;
    const u = 1 - t;
    const px = u * u * x1 + 2 * u * t * cx + t * t * x2;
    const py = u * u * y1 + 2 * u * t * cy + t * t * y2;

    const color = p.hue === "success" ? THEME.success : THEME.accent;
    const glow = ctx.createRadialGradient(px, py, 0, px, py, 6);
    glow.addColorStop(0, color);
    glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fill();
    return true;
  });
}

function drawNode(ctx, node, time, compact) {
  const { x, y, type, status, label, detail } = node;
  const active = status === "live" || status === "active" || status === "hybrid";
  const r = type === "feed" ? (compact ? 14 : 16) : type === "output" ? (compact ? 16 : 18) : compact ? 12 : 14;
  const pulse = active ? 1 + 0.08 * Math.sin(time * 3 + x * 0.05) : 1;

  if (active) {
    const glow = ctx.createRadialGradient(x, y, r * 0.2, x, y, r * 2.2 * pulse);
    glow.addColorStop(0, `${nodeColor(type, status)}44`);
    glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, r * 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  const fill = nodeColor(type, status);
  const grad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
  grad.addColorStop(0, "#ffffff22");
  grad.addColorStop(0.4, fill);
  grad.addColorStop(1, "#0f172a");

  ctx.fillStyle = grad;
  ctx.strokeStyle = active ? THEME.accentGlow : "rgba(148,163,184,0.35)";
  ctx.lineWidth = active ? 1.5 : 1;
  ctx.beginPath();
  ctx.arc(x, y, r * pulse, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = THEME.text;
  ctx.font = `600 ${compact ? 7 : 8}px 'DM Sans', system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const lines = String(label).split("\n");
  lines.forEach((line, i) => {
    ctx.fillText(line, x, y + (i - (lines.length - 1) / 2) * (compact ? 8 : 9));
  });

  if (type === "output" && detail) {
    ctx.fillStyle = THEME.success;
    ctx.font = `700 ${compact ? 8 : 9}px 'JetBrains Mono', monospace`;
    ctx.fillText(detail, x, y + r + (compact ? 10 : 12));
  } else if (type === "feature" && Number.isFinite(node.value)) {
    ctx.fillStyle = THEME.muted;
    ctx.font = `500 ${compact ? 6 : 7}px 'JetBrains Mono', monospace`;
    ctx.fillText(node.value.toFixed(2), x, y + r + 8);
  }
}

function drawColumnLabels(ctx, w, compact) {
  const labels = ["Ingest", "Features", "Encode", "Models", "Merge", "Output"];
  const xs = compact ? [0.06, 0.26, 0.44, 0.58, 0.72, 0.88] : [0.07, 0.24, 0.4, 0.56, 0.72, 0.9];
  ctx.fillStyle = THEME.muted;
  ctx.font = `500 ${compact ? 7 : 8}px 'DM Sans', system-ui, sans-serif`;
  ctx.textAlign = "center";
  xs.forEach((frac, i) => {
    ctx.fillText(labels[i], frac * w + (compact ? 20 : 36), 16);
  });
}

function drawLiveNetwork(canvas, flow, time, dt, hoveredEdgeIndex) {
  if (!flow) return;

  const compact = isCompactViewport(window.innerWidth);
  const minWidth = compact ? 720 : 920;
  const { ctx, w, h } = setupCanvas(canvas, { minWidth, height: compact ? 340 : 380 });
  const layout = buildLiveLayout(flow, w, h, compact);
  graphState = { ...layout, hoveredEdgeIndex };

  drawBackground(ctx, w, h, time);
  drawColumnLabels(ctx, w - (compact ? 40 : 72), compact);

  layout.edges.forEach((edge, index) => {
    drawEdge(ctx, edge, time, index === hoveredEdgeIndex);
  });

  drawParticles(ctx, layout.edges, dt);
  layout.nodes.forEach((node) => drawNode(ctx, node, time, compact));

  const spot = flow.spot?.name || "Monitoring spot";
  const alert = flow.alert?.title || "No active hazard";
  const inferMs = flow.inference_ms != null ? `${flow.inference_ms} ms` : "k-NN";
  const backend = flow.backend === "openvino" ? "OpenVINO" : flow.backend === "onnxruntime" ? "ONNX" : "local";

  ctx.fillStyle = THEME.text;
  ctx.font = `600 ${compact ? 10 : 11}px 'DM Sans', system-ui, sans-serif`;
  ctx.textAlign = "left";
  ctx.fillText(`Live · ${spot}`, 14, compact ? 28 : 32);

  ctx.fillStyle = THEME.muted;
  ctx.font = `400 ${compact ? 8 : 9}px 'DM Sans', system-ui, sans-serif`;
  const sub = compact ? alert.slice(0, 42) + (alert.length > 42 ? "…" : "") : alert;
  ctx.fillText(sub, 14, compact ? 40 : 46);
  ctx.textAlign = "right";
  ctx.fillText(`${backend} · ${inferMs}`, w - 14, compact ? 28 : 32);
}

function updateLiveBar(flow) {
  const bar = document.getElementById("evoLiveStatus");
  if (!bar || !flow) return;

  const steps = (flow.pipeline_status || [])
    .map((s) => `${s.label}: ${s.status}`)
    .join(" · ");
  const updated = flow.generated_at
    ? new Date(flow.generated_at).toLocaleTimeString()
    : "—";
  const mode = flow.prediction?.inference_mode || "hybrid";
  bar.textContent = `${steps} · ${mode} · updated ${updated}`;
}

function showEdgeTooltip(edge, clientX, clientY) {
  const tip = document.getElementById("evoEdgeTooltip");
  const scroll = document.getElementById("evoVizScroll");
  if (!tip || !scroll || !edge) return;

  tip.classList.remove("hidden");
  tip.innerHTML = `<strong>${edge.label}</strong><span>${edge.detail}</span>${
    Number.isFinite(edge.value) ? `<code>${Number(edge.value).toFixed(3)}</code>` : ""
  }`;

  const rect = scroll.getBoundingClientRect();
  const x = clientX - rect.left + scroll.scrollLeft;
  const y = clientY - rect.top;
  tip.style.left = `${Math.min(x + 12, scroll.scrollWidth - 220)}px`;
  tip.style.top = `${Math.max(y - 8, 8)}px`;
}

function hideEdgeTooltip() {
  document.getElementById("evoEdgeTooltip")?.classList.add("hidden");
}

function updateVizCaption() {
  const caption = document.getElementById("evoVizCaption");
  if (!caption) return;
  const compact = isCompactViewport(window.innerWidth);
  caption.textContent = compact
    ? "Live inference · tap a connection to inspect data"
    : "Live inference · hover connections to see data flowing through the model";
}

function drawLossChart(canvas, metrics) {
  const compact = isCompactViewport(window.innerWidth);
  const { ctx, w, h } = setupCanvas(canvas, { height: compact ? 150 : 180, fillParent: true });
  ctx.clearRect(0, 0, w, h);

  const train = metrics.train_loss || [];
  const val = metrics.val_loss || [];
  if (!train.length && !val.length) {
    ctx.fillStyle = THEME.muted;
    ctx.font = "12px 'DM Sans', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(metrics.message || "Training metrics pending", w / 2, h / 2);
    return;
  }

  const maxY = Math.max(...train, ...(val.length ? val : [0])) * 1.06 || 1;
  const pad = { t: 24, r: 12, b: 24, l: 32 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  ctx.strokeStyle = "rgba(148,163,184,0.14)";
  ctx.lineWidth = 1;
  roundRect(ctx, pad.l, pad.t, plotW, plotH, 4);
  ctx.stroke();

  const plotLine = (data, color) => {
    if (!data.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach((y, i) => {
      const x = pad.l + (i / Math.max(data.length - 1, 1)) * plotW;
      const yy = pad.t + plotH - (y / maxY) * plotH;
      if (i === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    });
    ctx.stroke();
  };

  plotLine(train, THEME.accent);
  plotLine(val, "rgba(148,163,184,0.85)");

  ctx.font = "500 9px 'DM Sans', system-ui, sans-serif";
  ctx.fillStyle = THEME.accent;
  ctx.textAlign = "left";
  ctx.fillText("Train", pad.l, 12);
  ctx.fillStyle = THEME.muted;
  ctx.fillText("Val", pad.l + 36, 12);
}

function renderMetrics(data) {
  const metrics = data.metrics || {};
  const format = (value, digits = 3) =>
    Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
  const gateStatus = metrics.all_quality_gates_pass;
  const approved = gateStatus === true;
  const backend = data.backend || liveFlow?.backend || "—";
  const runtimeKey = backend === "openvino" ? "openvino" : "onnx";
  const runtimeLabel =
    backend === "openvino" ? "OpenVINO" : backend === "onnxruntime" ? "ONNX Runtime" : backend;

  const el = document.getElementById("evoMetrics");
  if (!el) return;

  const livePred = liveFlow?.prediction || {};
  el.innerHTML = `
    <div class="evo-metrics-header">
      <span class="status-badge ${approved ? "approved" : "research"}">${approved ? "Gates passed" : "Research preview"}</span>
      <p class="evo-metrics-note">${glossaryTerm("hybrid", "Hybrid policy")} · live spot: <strong>${liveFlow?.spot?.name || "—"}</strong></p>
    </div>
    <div class="evo-metrics-grid">
      <div class="metric-row">
        <span class="metric-label">Live success</span>
        <span class="metric-value">${livePred.success_pct != null ? `${livePred.success_pct}%` : "—"}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Live time</span>
        <span class="metric-value">${livePred.time_min != null ? `${livePred.time_min} min` : "—"}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">${glossaryTerm("mae_success", "Success MAE")}</span>
        <span class="metric-value">${format(metrics.val_mae_success_pct)}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">${glossaryTerm("r2_success", "Success R²")}</span>
        <span class="metric-value">${format(metrics.val_r2_success_pct)}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">${glossaryTerm("mae_time", "Time MAE")}</span>
        <span class="metric-value">${format(metrics.val_mae_time_min)} min</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">${glossaryTerm(runtimeKey, "Runtime")}</span>
        <span class="metric-value">${runtimeLabel}</span>
      </div>
    </div>
    <p class="evo-metrics-footnote">
      ${glossaryTerm("dual_head", "Dual-head")} ensemble:
      ${glossaryTerm("mlp", "MLP")} + ${glossaryTerm("lightgbm", "boosting")}.
      Diagram shows real feed → feature → model flow from the current dashboard.
    </p>
  `;
  initGlossaryTooltips(el);
}

function renderEvoViz() {
  updateVizCaption();
  const lossCanvas = document.getElementById("evoLossCanvas");
  const lossTitle = document.getElementById("evoLossTitle");
  if (lossTitle && evoData) {
    lossTitle.innerHTML = `${glossaryTerm("train_loss", "Train")} / ${glossaryTerm("val_loss", "val")} loss`;
    initGlossaryTooltips(lossTitle);
  }
  if (lossCanvas && evoData) drawLossChart(lossCanvas, evoData.metrics || {});
  if (evoData) renderMetrics(evoData);
  if (liveFlow) updateLiveBar(liveFlow);
}

async function loadLiveFlow() {
  const data = await fetchJson("/api/evo/live-flow?use_evo=true");
  if (!data) return null;

  const stamp = `${data.generated_at}-${data.spot?.id}-${data.prediction?.success_pct}`;
  if (stamp !== lastFlowStamp) {
    lastFlowStamp = stamp;
    spawnInferenceWave(true);
  }
  liveFlow = data;
  updateLiveBar(data);
  if (evoData) renderMetrics(evoData);
  return data;
}

function startLivePolling() {
  stopLivePolling();
  loadLiveFlow();
  livePollTimer = window.setInterval(() => {
    if (modalOpen) loadLiveFlow();
  }, LIVE_POLL_MS);
}

function stopLivePolling() {
  if (livePollTimer) {
    clearInterval(livePollTimer);
    livePollTimer = null;
  }
}

let hoveredEdgeIndex = -1;
let lastFrameTs = 0;

function bindCanvasInteraction(canvas) {
  if (!canvas || canvas.dataset.evoBound) return;
  canvas.dataset.evoBound = "1";

  const pickEdge = (event) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.clientWidth / rect.width;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleX;
    const hit = findEdgeAt(x, y, isCompactViewport(window.innerWidth) ? 14 : 10);
    hoveredEdgeIndex = hit ? hit.index : -1;
    if (hit) showEdgeTooltip(hit.edge, event.clientX, event.clientY);
    else hideEdgeTooltip();
    canvas.style.cursor = hit ? "crosshair" : "default";
  };

  canvas.addEventListener("mousemove", pickEdge);
  canvas.addEventListener("mouseleave", () => {
    hoveredEdgeIndex = -1;
    hideEdgeTooltip();
    canvas.style.cursor = "default";
  });
  canvas.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches[0]) pickEdge(e.touches[0]);
    },
    { passive: true }
  );
}

function startNetworkAnimation() {
  const netCanvas = document.getElementById("evoNetworkCanvas");
  if (!netCanvas) return;
  bindCanvasInteraction(netCanvas);

  let start = null;
  const tick = (ts) => {
    if (!modalOpen) {
      animationFrame = null;
      return;
    }
    if (!start) start = ts;
    const time = (ts - start) / 1000;
    const dt = lastFrameTs ? (ts - lastFrameTs) / 1000 : 0.016;
    lastFrameTs = ts;

    if (liveFlow) {
      if (time - lastParticleSpawn > 1.4) {
        lastParticleSpawn = time;
        spawnInferenceWave(false);
      }
      drawLiveNetwork(netCanvas, liveFlow, time, dt, hoveredEdgeIndex);
    } else {
      const { ctx, w, h } = setupCanvas(netCanvas, { minWidth: 720, height: 380 });
      drawBackground(ctx, w, h, time);
      ctx.fillStyle = THEME.muted;
      ctx.font = "13px 'DM Sans', system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for live inference data…", w / 2, h / 2);
    }

    animationFrame = requestAnimationFrame(tick);
  };

  if (animationFrame) cancelAnimationFrame(animationFrame);
  lastFrameTs = 0;
  animationFrame = requestAnimationFrame(tick);
}

function stopNetworkAnimation() {
  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }
  particles = [];
  hoveredEdgeIndex = -1;
  hideEdgeTooltip();
}

function openEvoModal() {
  const modal = document.getElementById("evoModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  modalOpen = true;
  document.body.style.overflow = "hidden";
  document.body.classList.add("evo-modal-open");
  renderEvoViz();
  startLivePolling();
  startNetworkAnimation();
  const scroll = document.getElementById("evoVizScroll");
  if (scroll) scroll.scrollLeft = 0;
}

function closeEvoModal() {
  const modal = document.getElementById("evoModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modalOpen = false;
  document.body.style.overflow = "";
  document.body.classList.remove("evo-modal-open");
  stopLivePolling();
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
  const statusEl = document.getElementById("evoStatus");
  if (statusEl) {
    statusEl.innerHTML = data.available
      ? `${modelVersion} · ${backend} · ${glossaryTerm("hybrid", "hybrid")} · live diagram`
      : `${modelVersion} · awaiting model artifacts`;
    initGlossaryTooltips(statusEl);
  }
  initGlossaryTooltips(document.getElementById("evoModal"));
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

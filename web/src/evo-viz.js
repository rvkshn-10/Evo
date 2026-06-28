const API_BASE = import.meta.env.VITE_API_BASE || "";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok || !contentType.includes("application/json")) return null;
  return response.json();
}

function drawNetworkGraph(canvas, architecture) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const layers = architecture.layers || [];
  const edges = architecture.edges || [];
  const positions = {};
  const gap = w / (layers.length + 1);

  layers.forEach((layer, index) => {
    positions[layer.id] = { x: gap * (index + 1), y: h / 2 };
  });

  ctx.strokeStyle = "rgba(76, 201, 240, 0.45)";
  ctx.lineWidth = 2;
  edges.forEach(([from, to]) => {
    const a = positions[from];
    const b = positions[to];
    if (!a || !b) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  layers.forEach((layer) => {
    const p = positions[layer.id];
    if (!p) return;
    const color =
      layer.type === "input" ? "#4cc9f0" : layer.type === "output" ? "#ffd166" : "#ff6b6b";
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 28, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e8eef7";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    const lines = (layer.label || layer.id).split("\n");
    lines.forEach((line, i) => {
      ctx.fillText(line, p.x, p.y - 40 + i * 13);
    });
  });
}

function drawLossChart(canvas, metrics) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const train = metrics.train_loss || [];
  const val = metrics.val_loss || [];
  if (!train.length && !val.length) {
    ctx.fillStyle = "#9fb0c4";
    ctx.font = "13px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(metrics.message || "Training metrics pending", w / 2, h / 2);
    return;
  }

  const series = train.length ? train : val;
  const maxY = Math.max(...series, ...(val.length ? val : [0])) * 1.1 || 1;
  const pad = 24;

  const plot = (data, color) => {
    if (!data.length) return;
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

  plot(train, "#4cc9f0");
  plot(val, "#ffd166");

  ctx.fillStyle = "#9fb0c4";
  ctx.font = "11px system-ui";
  ctx.fillText("Train loss", pad, 14);
  ctx.fillStyle = "#ffd166";
  ctx.fillText("Val loss", pad + 70, 14);
}

export async function initEvoVisualization() {
  const panel = document.getElementById("evoPanel");
  const modeSelect = document.getElementById("runModeSelect");
  if (!panel) return;

  const updateVisibility = () => {
    const isEvo = modeSelect?.value === "evo";
    panel.classList.toggle("hidden", !isEvo);
  };
  modeSelect?.addEventListener("change", updateVisibility);

  const data = await fetchJson("/api/evo/visualization");
  if (!data) return;

  document.getElementById("evoStatus").textContent = data.available
    ? `Evo 1.0 · ${data.backend || "ready"}`
    : "Evo 1.0 · awaiting model files from GitHub";

  const netCanvas = document.getElementById("evoNetworkCanvas");
  const lossCanvas = document.getElementById("evoLossCanvas");
  if (netCanvas) drawNetworkGraph(netCanvas, data.architecture);
  if (lossCanvas) drawLossChart(lossCanvas, data.metrics);

  const mae = data.metrics || {};
  document.getElementById("evoMetrics").innerHTML = `
    <p><strong>Val MAE success:</strong> ${mae.val_mae_success_pct ?? "—"}%</p>
    <p><strong>Val MAE time:</strong> ${mae.val_mae_time_min ?? "—"} min</p>
    <p><strong>OpenVINO:</strong> ${data.openvino_connected ? "connected" : "not loaded"}</p>
  `;

  updateVisibility();
}

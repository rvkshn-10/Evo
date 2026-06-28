const API_BASE = import.meta.env.VITE_API_BASE || "";

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function getSemesterRange() {
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1;
  if (month >= 8) {
    return {
      since: `${year}-08-01T00:00:00Z`,
      until: `${year}-12-31T23:59:59Z`,
      label: `Fall ${year}`,
    };
  }
  if (month <= 5) {
    return {
      since: `${year}-01-01T00:00:00Z`,
      until: `${year}-05-31T23:59:59Z`,
      label: `Spring ${year}`,
    };
  }
  return {
    since: `${year}-01-01T00:00:00Z`,
    until: `${year}-05-31T23:59:59Z`,
    label: `Spring ${year}`,
  };
}

function getRangeFromPreset(preset) {
  const now = new Date();
  const until = now.toISOString();
  if (preset === "7d") {
    const since = new Date(now.getTime() - 7 * 86400000).toISOString();
    return { since, until, label: "Last 7 days" };
  }
  if (preset === "30d") {
    const since = new Date(now.getTime() - 30 * 86400000).toISOString();
    return { since, until, label: "Last 30 days" };
  }
  if (preset === "90d") {
    const since = new Date(now.getTime() - 90 * 86400000).toISOString();
    return { since, until, label: "Last 90 days" };
  }
  if (preset === "semester") {
    return getSemesterRange();
  }
  return { since: null, until: null, label: "All time" };
}

function rangeQuery(range) {
  const params = new URLSearchParams();
  if (range.since) params.set("since", range.since);
  if (range.until) params.set("until", range.until);
  const query = params.toString();
  return query ? `?${query}` : "";
}

function drawLineChart(canvas, series, keys, colors) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 420;
  const height = canvas.clientHeight || 180;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  if (!series.length) {
    ctx.fillStyle = "rgba(180, 190, 210, 0.7)";
    ctx.font = "13px DM Sans, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No runs yet — click Run Agent to start logging history", width / 2, height / 2);
    return;
  }

  const pad = { top: 12, right: 12, bottom: 24, left: 36 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const allValues = keys.flatMap((key) => series.map((row) => Number(row[key]) || 0));
  const maxY = Math.max(1, ...allValues);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  keys.forEach((key, keyIndex) => {
    const color = colors[keyIndex] || "#4cc9f0";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((row, index) => {
      const x = pad.left + (plotW * index) / Math.max(1, series.length - 1);
      const value = Number(row[key]) || 0;
      const y = pad.top + plotH - (value / maxY) * plotH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  ctx.fillStyle = "rgba(180, 190, 210, 0.65)";
  ctx.font = "10px DM Sans, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(String(maxY), pad.left - 4, pad.top + 4);
  ctx.fillText("0", pad.left - 4, height - pad.bottom);
}

function formatWhen(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

function renderHighRiskTable(rows) {
  const body = document.getElementById("historyHighRiskBody");
  if (!body) return;
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="6" class="subtitle">No high-risk predictions in this range.</td></tr>`;
    return;
  }
  body.innerHTML = rows
    .slice(0, 50)
    .map(
      (row) => `
      <tr>
        <td>${row.spot_name || row.spot_id || "—"}</td>
        <td>${row.event_type || "—"}</td>
        <td>${row.occupancy ?? "—"}</td>
        <td>${row.predicted_evacuation_rate != null ? `${row.predicted_evacuation_rate}%` : "—"}</td>
        <td><span class="risk-${row.risk_level || "low"}">${row.risk_level || "—"}</span></td>
        <td>${formatWhen(row.recorded_at)}</td>
      </tr>`,
    )
    .join("");
}

async function loadHistoryPanel() {
  const preset = document.getElementById("historyRangeSelect")?.value || "30d";
  const range = getRangeFromPreset(preset);
  const query = rangeQuery(range);

  const [meta, timeseries, highRisk] = await Promise.all([
    fetch(apiUrl("/api/history?limit=5")).then((r) => r.json()).catch(() => null),
    fetch(apiUrl(`/api/history/timeseries${query}`)).then((r) => r.json()).catch(() => null),
    fetch(apiUrl(`/api/history/high-risk${query}&risk_level=high,medium`))
      .then((r) => r.json())
      .catch(() => null),
  ]);

  const storageEl = document.getElementById("historyStorageNote");
  if (storageEl && meta?.storage) {
    const backend = meta.storage.backend === "postgres" ? "Neon Postgres" : "Local SQLite";
    storageEl.textContent = `${backend} · ${meta.total_snapshots ?? 0} total runs · showing ${range.label}`;
  }

  const points = timeseries?.points || [];
  drawLineChart(
    document.getElementById("historyRiskChart"),
    points,
    ["high_risk_spots"],
    ["#ff6b6b"],
  );
  drawLineChart(
    document.getElementById("historyAlertsChart"),
    points,
    ["active_alerts", "significant_earthquakes"],
    ["#4cc9f0", "#ffd166"],
  );

  renderHighRiskTable(highRisk?.predictions || []);
}

function downloadExport(format) {
  const preset = document.getElementById("historyRangeSelect")?.value || "30d";
  const range = getRangeFromPreset(preset);
  const params = new URLSearchParams({ format });
  if (range.since) params.set("since", range.since);
  if (range.until) params.set("until", range.until);
  window.open(apiUrl(`/api/history/export?${params}`), "_blank");
}

function openHistoryModal() {
  const modal = document.getElementById("historyModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  document.body.classList.add("history-modal-open");
  loadHistoryPanel();
}

function closeHistoryModal() {
  const modal = document.getElementById("historyModal");
  if (!modal) return;
  modal.classList.add("hidden");
  document.body.style.overflow = "";
  document.body.classList.remove("history-modal-open");
}

export function initHistoryPanel() {
  document.getElementById("historyBtn")?.addEventListener("click", openHistoryModal);
  document.getElementById("historyModalClose")?.addEventListener("click", closeHistoryModal);
  document.getElementById("historyModalBackdrop")?.addEventListener("click", closeHistoryModal);
  document.getElementById("historyRangeSelect")?.addEventListener("change", loadHistoryPanel);
  document.getElementById("exportJsonBtn")?.addEventListener("click", () => downloadExport("json"));
  document.getElementById("exportCsvBtn")?.addEventListener("click", () => downloadExport("csv"));
  document.getElementById("exportSqliteBtn")?.addEventListener("click", () => downloadExport("sqlite"));
}

export async function refreshHistoryIfOpen() {
  const modal = document.getElementById("historyModal");
  if (modal && !modal.classList.contains("hidden")) {
    await loadHistoryPanel();
  }
}

let map;
let markersLayer;
let heatLayer = null;
let pipelinePollTimer = null;
let lastSnapshot = null;
let notificationsEnabled = false;

const API_BASE = import.meta.env.VITE_API_BASE || "";

const HAZARD_FILTERS = [
  { id: "earthquake", label: "Earthquakes", icon: "🫨", default: true },
  { id: "wildfire", label: "Wildfires / Fire", icon: "🔥", default: true },
  { id: "flood", label: "Floods", icon: "🌊", default: true },
  { id: "tornado", label: "Tornadoes", icon: "🌪️", default: true },
  { id: "tsunami", label: "Tsunamis", icon: "🌊", default: true },
  { id: "severe_weather", label: "Severe Weather", icon: "⛈️", default: true },
  { id: "monitoring", label: "Monitoring Spots", icon: "📍", default: true },
];

const activeFilters = Object.fromEntries(HAZARD_FILTERS.map((f) => [f.id, f.default]));
let showHeatmap = true;
let openvinoRuntimeCache = null;

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok || !contentType.includes("application/json")) {
    return null;
  }
  return response.json();
}

function isFilterActive(category) {
  return activeFilters[category] !== false;
}

function setConnectionBadge(mode) {
  const el = document.getElementById("connectionBadge");
  if (mode === "live") {
    el.textContent = "API: Live";
    el.className = "badge badge-live";
  } else if (mode === "demo") {
    el.textContent = "API: Demo data";
    el.className = "badge badge-demo";
  } else {
    el.textContent = "API: Offline";
    el.className = "badge badge-offline";
  }
}

async function fetchDashboard() {
  const mode = document.getElementById("runModeSelect")?.value || "sync";
  const params = new URLSearchParams();
  if (mode === "evo") params.set("use_evo", "true");
  if (mode === "evo13") params.set("use_evo13", "true");
  const query = params.toString();
  const path = query ? `/api/dashboard?${query}` : "/api/dashboard";

  try {
    const data = await fetchJson(apiUrl(path));
    if (data) {
      setConnectionBadge("live");
      return data;
    }
  } catch (error) {
    console.warn("Live API unavailable, using demo snapshot", error);
  }

  try {
    const data = await fetchJson("/demo-snapshot.json");
    if (data) {
      setConnectionBadge("demo");
      return data;
    }
  } catch (error) {
    console.error("Demo snapshot unavailable", error);
  }

  setConnectionBadge("offline");
  throw new Error("Dashboard unavailable");
}

async function fetchPipelineStatus() {
  try {
    const data = await fetchJson(apiUrl("/api/pipeline/status"));
    return data || { status: "idle" };
  } catch {
    return { status: "idle" };
  }
}

function showToast(title, message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<strong>${title}</strong><p>${message}</p>`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 8000);
}

function pushDeviceNotification(title, message, options = {}) {
  if (!notificationsEnabled || Notification.permission !== "granted") return false;
  try {
    new Notification(title, {
      body: message,
      tag: options.tag || "emergency-agent-run",
      icon: options.icon,
    });
    return true;
  } catch (error) {
    console.warn("Device notification failed", error);
    return false;
  }
}

/** Agent run alerts: device push when Notify is on, otherwise in-page toast. */
function notifyAgentEvent(title, message, type = "info") {
  if (pushDeviceNotification(title, message)) return;
  showToast(title, message, type);
}

function summarizeAgentSnapshot(snapshot) {
  const alerts = (snapshot.alerts || []).length + (snapshot.earthquakes || []).length;
  const risk = snapshot.summary?.high_risk_spots ?? 0;
  const ps =
    snapshot.peoplesense_mode === "live"
      ? snapshot.peoplesense_source === "get_api"
        ? "PeopleSense live"
        : "PeopleSense live"
      : "PeopleSense simulated";
  const mode = document.getElementById("runModeSelect")?.value || "sync";
  return `${alerts} hazards · ${risk} high-risk spots · ${ps} · mode: ${mode}`;
}

async function enableNotifications() {
  if (!("Notification" in window)) {
    showToast("Notifications blocked", "Your browser does not support notifications.", "warn");
    return;
  }
  const permission = await Notification.requestPermission();
  notificationsEnabled = permission === "granted";
  document.getElementById("notifyBtn").textContent = notificationsEnabled ? "🔔 On" : "🔔 Off";
  if (notificationsEnabled) {
    showToast(
      "Device notifications on",
      "Run Agent will alert this device when the pipeline finishes.",
      "ok",
    );
  }
}

function isEvoRunMode() {
  const mode = document.getElementById("runModeSelect")?.value;
  return mode === "evo" || mode === "evo13";
}

function isEvo12RunMode() {
  return document.getElementById("runModeSelect")?.value === "evo";
}

function isEvo13RunMode() {
  return document.getElementById("runModeSelect")?.value === "evo13";
}

function isExternalAiMode() {
  return document.getElementById("runModeSelect")?.value === "external_ai";
}

function updateExternalAiWarning() {
  updateRunModeBanners();
}

function updateRunModeBanners() {
  const external = document.getElementById("externalAiWarning");
  const evo = document.getElementById("evoLocalNote");
  const evo13 = document.getElementById("evo13ResearchWarning");
  if (external) external.classList.toggle("hidden", !isExternalAiMode());
  if (evo) evo.classList.toggle("hidden", !isEvo12RunMode());
  if (evo13) evo13.classList.toggle("hidden", !isEvo13RunMode());
  if (isEvo12RunMode() && evo) {
    import("./glossary.js").then(({ initGlossaryTooltips }) => initGlossaryTooltips(evo));
  }
}

function warnEvo13Research({ onRun = false } = {}) {
  if (!isEvo13RunMode()) return;
  const title = onRun ? "Starting Evo 1.3 research run" : "Evo 1.3 research selected";
  const message = onRun
    ? "Predictions use internet hazard data and public studies — not FCUSD drill validation. May not be accurate."
    : "Research-only mode. Estimates all sites from enriched reference + live hazards. Use Evo 1.2 hybrid for production.";
  showToast(title, message, "warn");
function warnExternalAiCredits({ onRun = false } = {}) {
  if (!isExternalAiMode()) return;
  const title = onRun ? "Starting External AI run" : "External AI selected";
  const message = onRun
    ? "This run will call Gemini (then OpenAI if needed). API credits apply — use Sync or Evo to avoid charges."
    : "Each Run Agent in this mode uses Gemini/OpenAI. Credits will run out over time — use Sync or Evo for free updates.";
  showToast(title, message, "warn");
}

function updateOpenvinoRowVisibility() {
  const row = document.getElementById("openvinoRow");
  if (!row) return;
  row.classList.toggle("hidden", !isEvo12RunMode());
}

function setOpenvinoDot(state) {
  const dot = document.getElementById("openvinoStatusDot");
  if (!dot) return;
  dot.classList.remove("connected", "fallback");
  if (state === "connected") dot.classList.add("connected");
  else if (state === "fallback") dot.classList.add("fallback");
}

async function fetchOpenvinoRuntime() {
  const data = await fetchJson(apiUrl("/api/evo/runtime"));
  if (data) openvinoRuntimeCache = data;
  return data;
}

function openOpenvinoModal(status) {
  const modal = document.getElementById("openvinoModal");
  const statusEl = document.getElementById("openvinoModalStatus");
  if (!modal || !statusEl) return;

  if (status?.openvino_connected) {
    statusEl.textContent = `Connected — Evo is using OpenVINO (${status.model_version || "evo1.2"}).`;
  } else if (status?.backend === "onnxruntime") {
    statusEl.textContent =
      `Not connected — Evo is running on ONNX Runtime (CPU). OpenVINO is installed in settings but not active, or IR files are missing.`;
  } else if (!status?.available) {
    statusEl.textContent = "Evo model files not found on the API server. Ensure models/evo1.2/ exists and restart python3 main.py.";
  } else {
    statusEl.textContent =
      "OpenVINO is not connected. Follow the steps below, restart the API, then click Check again.";
  }

  modal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeOpenvinoModal() {
  const modal = document.getElementById("openvinoModal");
  if (!modal) return;
  modal.classList.add("hidden");
  document.body.style.overflow = "";
}

async function syncOpenvinoUi({ showGuideIfDisconnected = false } = {}) {
  updateOpenvinoRowVisibility();
  if (!isEvo12RunMode()) return;

  const status = await fetchOpenvinoRuntime();
  const toggle = document.getElementById("openvinoToggle");
  if (!status) {
    setOpenvinoDot("unknown");
    if (toggle) toggle.checked = false;
    return;
  }

  if (status.openvino_connected) {
    setOpenvinoDot("connected");
    if (toggle) toggle.checked = true;
  } else if (status.loaded && status.backend === "onnxruntime") {
    setOpenvinoDot("fallback");
    if (toggle) toggle.checked = false;
  } else {
    setOpenvinoDot("unknown");
    if (toggle) toggle.checked = false;
  }

  if (showGuideIfDisconnected && !status.openvino_connected) {
    openOpenvinoModal(status);
  }
}

async function onOpenvinoLabelClick() {
  const status = await fetchOpenvinoRuntime();
  if (status?.openvino_connected) {
    setOpenvinoDot("connected");
    const toggle = document.getElementById("openvinoToggle");
    if (toggle) toggle.checked = true;
  } else if (status?.backend === "onnxruntime") {
    setOpenvinoDot("fallback");
  }
  openOpenvinoModal(status);
}

async function onOpenvinoToggleChange(event) {
  const checkbox = event.target;
  if (!checkbox.checked) return;

  const status = await fetchOpenvinoRuntime();
  if (status?.openvino_connected) {
    setOpenvinoDot("connected");
    showToast("OpenVINO connected", `Evo inference via OpenVINO (${status.model_version}).`, "ok");
    return;
  }

  checkbox.checked = false;
  setOpenvinoDot(status?.backend === "onnxruntime" ? "fallback" : "unknown");
  openOpenvinoModal(status);
}

function initOpenvinoControls() {
  document.getElementById("openvinoToggle")?.addEventListener("change", onOpenvinoToggleChange);
  document.getElementById("openvinoLabelText")?.addEventListener("click", onOpenvinoLabelClick);
  document.getElementById("openvinoModalClose")?.addEventListener("click", closeOpenvinoModal);
  document.getElementById("openvinoModalDismiss")?.addEventListener("click", closeOpenvinoModal);
  document.getElementById("openvinoModalBackdrop")?.addEventListener("click", closeOpenvinoModal);
  document.getElementById("openvinoRecheckBtn")?.addEventListener("click", async () => {
    const status = await fetchOpenvinoRuntime();
    if (status?.openvino_connected) {
      closeOpenvinoModal();
      document.getElementById("openvinoToggle").checked = true;
      setOpenvinoDot("connected");
      showToast("OpenVINO connected", "Evo is now using OpenVINO.", "ok");
    } else {
      openOpenvinoModal(status);
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !document.getElementById("openvinoModal")?.classList.contains("hidden")) {
      closeOpenvinoModal();
    }
  });
}

async function notifyNewHazards(snapshot) {
  if (!lastSnapshot) return;

  const prevQuakeIds = new Set((lastSnapshot.earthquakes || []).map((q) => q.id));
  for (const quake of snapshot.earthquakes || []) {
    if (!prevQuakeIds.has(quake.id) && (quake.magnitude || 0) >= 4.5) {
      showToast(
        "Earthquake detected",
        quake.headline || `M${quake.magnitude} earthquake`,
        "warn"
      );
    }
  }
}

function riskClass(level) {
  return `risk ${level || "medium"}`;
}

function formatEta(seconds) {
  if (seconds == null || seconds <= 0) return "Almost done…";
  if (seconds < 60) return `About ${seconds}s remaining`;
  return `About ${Math.ceil(seconds / 60)} min remaining`;
}

function showPipelineProgress(show) {
  document.getElementById("pipelineProgress").classList.toggle("hidden", !show);
}

function renderPipelineStatus(status) {
  document.getElementById("progressBar").style.width = `${status.progress_percent ?? 0}%`;
  document.getElementById("pipelinePercent").textContent = `${status.progress_percent ?? 0}%`;
  document.getElementById("pipelineStepLabel").textContent =
    status.current_step_label || status.event_title || "Pipeline running…";
  document.getElementById("pipelineEta").textContent = formatEta(status.estimated_seconds_remaining);
}

function stopPipelinePolling() {
  if (pipelinePollTimer) {
    clearInterval(pipelinePollTimer);
    pipelinePollTimer = null;
  }
}

function startPipelinePolling(onComplete) {
  stopPipelinePolling();
  showPipelineProgress(true);
  const poll = async () => {
    try {
      const status = await fetchPipelineStatus();
      renderPipelineStatus(status);
      if (status.status === "completed") {
        stopPipelinePolling();
        showPipelineProgress(false);
        await onComplete?.(status);
      } else if (status.status === "failed") {
        stopPipelinePolling();
        showPipelineProgress(false);
        notifyAgentEvent("Agent run failed", status.error || "Unknown error", "warn");
        await onComplete?.(status);
      }
    } catch (error) {
      console.error(error);
    }
  };
  poll();
  pipelinePollTimer = setInterval(poll, 2000);
}

function getFilteredAlerts(snapshot) {
  return (snapshot.alerts || []).filter((a) =>
    isFilterActive(a.hazard_category || "severe_weather")
  );
}

function getFilteredEarthquakes(snapshot) {
  if (!isFilterActive("earthquake")) return [];
  return snapshot.earthquakes || [];
}

function renderFilters() {
  const container = document.getElementById("filterList");
  container.innerHTML = HAZARD_FILTERS.map(
    (f) => `
    <label class="filter-row">
      <input type="checkbox" data-filter="${f.id}" ${activeFilters[f.id] ? "checked" : ""} />
      <span>${f.icon} ${f.label}</span>
    </label>`
  ).join("");

  container.querySelectorAll("input[data-filter]").forEach((input) => {
    input.addEventListener("change", () => {
      activeFilters[input.dataset.filter] = input.checked;
      if (lastSnapshot) renderSnapshot(lastSnapshot);
    });
  });
}

function renderAlerts(snapshot) {
  const container = document.getElementById("alertsList");
  const quakes = getFilteredEarthquakes(snapshot);
  const alerts = getFilteredAlerts(snapshot);

  const items = [
    ...quakes.map((q) => ({
      event: "Earthquake",
      headline: q.headline || q.title,
      severity: q.severity || "Severe",
      expires: q.sent || q.time || "—",
      isEarthquake: true,
      magnitude: parseMagnitude(q),
    })),
    ...alerts.map((a) => ({ ...a, isEarthquake: false })),
  ];

  if (!items.length) {
    container.innerHTML = "<p class='subtitle'>No hazards match the selected filters.</p>";
    return;
  }

  container.innerHTML = items
    .map(
      (alert) => `
      <article class="alert-card ${alert.isEarthquake ? "earthquake-card" : ""}">
        <h4>${alert.event || "Alert"}${alert.isEarthquake ? " 🫨" : ""}</h4>
        <p>${alert.headline || alert.area_desc || ""}</p>
        <p><strong>Severity:</strong> ${alert.severity || "Unknown"}
        ${alert.magnitude ? ` · <strong>Mag:</strong> ${alert.magnitude}` : ""}
        · <strong>Time:</strong> ${alert.expires || alert.sent || "—"}</p>
      </article>`
    )
    .join("");
}

function renderPredictions(snapshot) {
  const rows = [];
  const alerts = [...getFilteredEarthquakes(snapshot), ...getFilteredAlerts(snapshot)];

  alerts.forEach((alert) => {
    (alert.evacuation_predictions || []).forEach((prediction) => {
      rows.push(`
        <tr>
          <td>${prediction.name}</td>
          <td>${alert.event || alert.headline || "Alert"}</td>
          <td>${prediction.inputs?.occupancy ?? "—"}</td>
          <td>${(prediction.predicted_evacuation_rate * 100).toFixed(1)}%</td>
          <td>${prediction.predicted_evacuation_time_min}</td>
          <td><span class="${riskClass(prediction.risk_level)}">${prediction.risk_level}</span></td>
          <td><code>${prediction.model || "—"}</code></td>
        </tr>
      `);
    });
  });

  document.getElementById("predictionsBody").innerHTML =
    rows.join("") || "<tr><td colspan='7'>No predictions for selected filters.</td></tr>";
}

function parseMagnitude(quake) {
  if (quake.magnitude != null) return quake.magnitude;
  const text = quake.headline || quake.title || "";
  const match = text.match(/M(\d+(?:\.\d+)?)/i);
  return match ? parseFloat(match[1]) : 4;
}

function getHeatmapPoints(snapshot) {
  if (snapshot.heatmap_points?.length) {
    return snapshot.heatmap_points;
  }

  return (snapshot.earthquakes || [])
    .filter((q) => q.center_lat != null && q.center_lon != null)
    .map((q) => {
      const mag = parseMagnitude(q);
      return {
        lat: q.center_lat,
        lon: q.center_lon,
        intensity: Math.min(1, mag / 8),
        hazard_category: "earthquake",
        label: q.headline || q.title || "Earthquake",
        magnitude: mag,
      };
    });
}

function renderHeatmap(snapshot) {
  if (heatLayer) {
    map.removeLayer(heatLayer);
    heatLayer = null;
  }
  if (!showHeatmap || !isFilterActive("earthquake")) return;

  const points = getHeatmapPoints(snapshot)
    .filter((p) => p.hazard_category === "earthquake" && isFilterActive("earthquake"))
    .map((p) => [p.lat, p.lon, Math.max(0.3, p.intensity || 0.5)]);

  if (points.length && window.L?.heatLayer) {
    heatLayer = L.heatLayer(points, {
      radius: 45,
      blur: 28,
      maxZoom: 10,
      gradient: { 0.2: "#1a535c", 0.4: "#ffe66d", 0.7: "#ff6b35", 1.0: "#d00000" },
    }).addTo(map);
  }
}

function renderMap(snapshot) {
  const quakes = getFilteredEarthquakes(snapshot);
  const alerts = getFilteredAlerts(snapshot);
  const center =
    quakes[0]?.center_lat != null
      ? { lat: quakes[0].center_lat, lon: quakes[0].center_lon }
      : snapshot.map_center || { lat: 38.58, lon: -121.3 };

  if (!map) {
    map = L.map("map").setView([center.lat, center.lon], quakes.length ? 7 : 6);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap",
    }).addTo(map);
    markersLayer = L.layerGroup().addTo(map);
  } else if (quakes.length) {
    map.setView([center.lat, center.lon], 7);
  }

  markersLayer.clearLayers();
  renderHeatmap(snapshot);

  if (isFilterActive("monitoring")) {
    (snapshot.monitoring_spots || []).forEach((spot) => {
      L.circleMarker([spot.lat, spot.lon], {
        radius: 8,
        color: "#4cc9f0",
        fillColor: "#4cc9f0",
        fillOpacity: 0.85,
      })
        .bindPopup(`<strong>${spot.name}</strong><br/>${spot.category}`)
        .addTo(markersLayer);
    });
  }

  alerts.forEach((alert) => {
    if (alert.center_lat == null || alert.center_lon == null) return;
    const color = alert.hazard_category === "wildfire" ? "#ff6b35" : "#ff6b6b";
    L.circle([alert.center_lat, alert.center_lon], {
      color,
      fillColor: color,
      fillOpacity: 0.15,
      radius: 8000,
    })
      .bindPopup(`<strong>${alert.event}</strong><br/>${alert.headline || ""}`)
      .addTo(markersLayer);
  });

  quakes.forEach((quake) => {
    if (quake.center_lat == null || quake.center_lon == null) return;
    const mag = parseMagnitude(quake);
    L.circleMarker([quake.center_lat, quake.center_lon], {
      radius: 10 + mag * 2,
      color: "#ffd166",
      fillColor: "#ffd166",
      fillOpacity: 0.95,
      weight: 2,
    })
      .bindPopup(
        `<strong>🫨 ${quake.headline || quake.title || "Earthquake"}</strong><br/>Magnitude: ${mag}`
      )
      .addTo(markersLayer);

    L.circle([quake.center_lat, quake.center_lon], {
      color: "#ffd166",
      fillColor: "#ffd166",
      fillOpacity: 0.08,
      radius: Math.max(15000, mag * 8000),
    }).addTo(markersLayer);
  });
}

function renderSnapshot(snapshot) {
  notifyNewHazards(snapshot);
  lastSnapshot = snapshot;

  const quakes = getFilteredEarthquakes(snapshot);
  const alerts = getFilteredAlerts(snapshot);

  document.getElementById("alertCount").textContent = quakes.length + alerts.length;
  document.getElementById("quakeCount").textContent = quakes.length;
  document.getElementById("riskCount").textContent = snapshot.summary?.high_risk_spots ?? 0;
  document.getElementById("modeBadge").textContent = (() => {
    const runMode = document.getElementById("runModeSelect")?.value || "sync";
    if (runMode === "evo13" || snapshot.prediction_policy === "evo1.3_research") {
      return "Evo 1.3 · Research";
    }
    if (runMode === "evo" || snapshot.prediction_policy === "evo1.2_hybrid") {
      return "Evo 1.2 · Production";
    }
    if (snapshot.peoplesense_mode === "live") {
      return snapshot.peoplesense_source === "get_api"
        ? "PeopleSense: Live (GET)"
        : "PeopleSense: Live";
    }
    return "PeopleSense: Placeholder";
  })();
  document.getElementById("lastUpdated").textContent =
    `Last updated: ${new Date(snapshot.generated_at).toLocaleString()}`;

  renderAlerts(snapshot);
  renderPredictions(snapshot);
  renderMap(snapshot);
}

async function finishAgentRun(status) {
  const snapshot = await fetchDashboard();
  renderSnapshot(snapshot);
  if (status?.status === "completed") {
    notifyAgentEvent("Agent run complete", summarizeAgentSnapshot(snapshot), "ok");
    const { refreshHistoryIfOpen } = await import("./history.js");
    refreshHistoryIfOpen();
  }
  return snapshot;
}

async function refresh() {
  const button = document.getElementById("refreshBtn");
  button.disabled = true;
  button.textContent = "Starting…";

  try {
    const mode = document.getElementById("runModeSelect")?.value || "sync";
    if (mode === "external_ai") {
      warnExternalAiCredits({ onRun: true });
    }
    if (mode === "evo13") {
      warnEvo13Research({ onRun: true });
    }
    const sync = await fetch(apiUrl(`/api/alerts/sync?mode=${encodeURIComponent(mode)}`), {
      method: "POST",
    });
    const info = await sync.json();

    if (info.status === "already_running") {
      startPipelinePolling(async (status) => {
        button.disabled = false;
        button.textContent = "Run Agent";
        await finishAgentRun(status);
      });
      return;
    }

    if (!sync.ok && sync.status !== 0) {
      showToast("Demo mode", "Pipeline requires local API. Showing latest data.", "warn");
      renderSnapshot(await fetchDashboard());
      button.disabled = false;
      button.textContent = "Run Agent";
      return;
    }

    button.textContent = "Pipeline running…";
    startPipelinePolling(async (status) => {
      try {
        await finishAgentRun(status);
      } finally {
        button.disabled = false;
        button.textContent = "Run Agent";
      }
    });
  } catch (error) {
    console.error(error);
    showToast("Could not start pipeline", "Use local server: python3 main.py", "warn");
    try {
      renderSnapshot(await fetchDashboard());
    } catch {
      document.getElementById("alertsList").innerHTML =
        "<p class='subtitle'>Start API: python3 main.py — then refresh.</p>";
    }
    button.disabled = false;
    button.textContent = "Run Agent";
    showPipelineProgress(false);
  }
}

export async function initApp() {
  renderFilters();

  document.getElementById("heatmapToggle").addEventListener("change", (e) => {
    showHeatmap = e.target.checked;
    if (lastSnapshot) renderMap(lastSnapshot);
  });

  document.getElementById("refreshBtn").addEventListener("click", refresh);
  document.getElementById("notifyBtn").addEventListener("click", enableNotifications);
  document.getElementById("runModeSelect")?.addEventListener("change", async () => {
    updateOpenvinoRowVisibility();
    updateExternalAiWarning();
    if (isExternalAiMode()) {
      warnExternalAiCredits();
    }
    if (isEvo13RunMode()) {
      warnEvo13Research();
    }
    if (isEvo12RunMode()) {
      await syncOpenvinoUi();
    }
    try {
      renderSnapshot(await fetchDashboard());
    } catch (error) {
      console.error(error);
    }
  });

  initOpenvinoControls();
  updateOpenvinoRowVisibility();
  updateExternalAiWarning();

  try {
    const [snapshot, pipelineStatus] = await Promise.all([
      fetchDashboard(),
      fetchPipelineStatus(),
    ]);

    renderSnapshot(snapshot);
    if (isEvo12RunMode()) {
      await syncOpenvinoUi();
    }

    const mendocino = (snapshot.earthquakes || []).find((q) => {
      const text = (q.headline || q.title || "").toLowerCase();
      return text.includes("redwood") || text.includes("mendocino");
    });
    if (mendocino) {
      showToast(
        "Northern CA earthquake",
        mendocino.headline || mendocino.title,
        "warn"
      );
    }

    if (pipelineStatus.status === "running") {
      const button = document.getElementById("refreshBtn");
      button.disabled = true;
      button.textContent = "Pipeline running…";
      startPipelinePolling(async (status) => {
        button.disabled = false;
        button.textContent = "Run Agent";
        await finishAgentRun(status);
      });
    }
  } catch (error) {
    console.error(error);
    document.getElementById("alertsList").innerHTML =
      "<p class='subtitle'>Start the API: <code>python3 main.py</code> then open http://localhost:8092</p>";
  }
}

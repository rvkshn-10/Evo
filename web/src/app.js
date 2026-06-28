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
  try {
    const data = await fetchJson(apiUrl("/api/dashboard"));
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

  if (notificationsEnabled && Notification.permission === "granted") {
    new Notification(title, { body: message });
  }
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
    showToast("Notifications enabled", "You will be alerted for significant hazards.", "ok");
  }
}

function notifyNewHazards(snapshot) {
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
        showToast("Pipeline complete", "Reports and broadcast files are ready.", "ok");
        onComplete?.(status);
      } else if (status.status === "failed") {
        stopPipelinePolling();
        showPipelineProgress(false);
        showToast("Pipeline failed", status.error || "Unknown error", "warn");
        onComplete?.(status);
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
        </tr>
      `);
    });
  });

  document.getElementById("predictionsBody").innerHTML =
    rows.join("") || "<tr><td colspan='6'>No predictions for selected filters.</td></tr>";
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
  document.getElementById("modeBadge").textContent =
    snapshot.peoplesense_mode === "live" ? "PeopleSense: Live" : "PeopleSense: Placeholder";
  document.getElementById("lastUpdated").textContent =
    `Last updated: ${new Date(snapshot.generated_at).toLocaleString()}`;

  renderAlerts(snapshot);
  renderPredictions(snapshot);
  renderMap(snapshot);
}

async function refresh() {
  const button = document.getElementById("refreshBtn");
  button.disabled = true;
  button.textContent = "Starting…";

  try {
    const mode = document.getElementById("runModeSelect")?.value || "sync";
    const sync = await fetch(apiUrl(`/api/alerts/sync?mode=${encodeURIComponent(mode)}`), {
      method: "POST",
    });
    const info = await sync.json();

    if (info.status === "already_running") {
      startPipelinePolling(async () => {
        button.disabled = false;
        button.textContent = "Run Agent";
        renderSnapshot(await fetchDashboard());
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
    showToast("Agent started", "Full emergency pipeline is running.", "info");
    startPipelinePolling(async () => {
      try {
        renderSnapshot(await fetchDashboard());
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

  try {
    const [snapshot, pipelineStatus] = await Promise.all([
      fetchDashboard(),
      fetchPipelineStatus(),
    ]);

    renderSnapshot(snapshot);

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
      startPipelinePolling(async () => {
        button.disabled = false;
        button.textContent = "Run Agent";
        renderSnapshot(await fetchDashboard());
      });
    }
  } catch (error) {
    console.error(error);
    document.getElementById("alertsList").innerHTML =
      "<p class='subtitle'>Start the API: <code>python3 main.py</code> then open http://localhost:8092</p>";
  }
}

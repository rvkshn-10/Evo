let map;
let markersLayer;
let pipelinePollTimer = null;

const API_BASE = import.meta.env.VITE_API_BASE || "";

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function fetchDashboard() {
  const response = await fetch(apiUrl("/api/dashboard"));
  if (!response.ok) throw new Error("Failed to load dashboard");
  return response.json();
}

async function fetchPipelineStatus() {
  const response = await fetch(apiUrl("/api/pipeline/status"));
  if (!response.ok) throw new Error("Failed to load pipeline status");
  return response.json();
}

function riskClass(level) {
  return `risk ${level || "medium"}`;
}

function formatEta(seconds) {
  if (seconds == null || seconds <= 0) return "Almost done…";
  if (seconds < 60) return `About ${seconds}s remaining`;
  const minutes = Math.ceil(seconds / 60);
  return `About ${minutes} min remaining`;
}

function showPipelineProgress(show) {
  document.getElementById("pipelineProgress").classList.toggle("hidden", !show);
}

function renderPipelineStatus(status) {
  const progress = status.progress_percent ?? 0;
  document.getElementById("progressBar").style.width = `${progress}%`;
  document.getElementById("pipelinePercent").textContent = `${progress}%`;
  document.getElementById("pipelineStepLabel").textContent =
    status.current_step_label || status.event_title || "Pipeline running…";
  document.getElementById("pipelineEta").textContent = formatEta(
    status.estimated_seconds_remaining
  );
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
        onComplete?.(status);
      } else if (status.status === "failed") {
        stopPipelinePolling();
        showPipelineProgress(false);
        alert(`Pipeline failed: ${status.error || "Unknown error"}`);
        onComplete?.(status);
      }
    } catch (error) {
      console.error(error);
    }
  };

  poll();
  pipelinePollTimer = setInterval(poll, 2000);
}

function renderAlerts(alerts, earthquakes = []) {
  const container = document.getElementById("alertsList");
  const items = [
    ...earthquakes.map((q) => ({
      event: q.event || "Earthquake",
      headline: q.headline || q.title,
      severity: q.severity || "Severe",
      expires: q.sent || "—",
      isEarthquake: true,
    })),
    ...alerts.map((a) => ({ ...a, isEarthquake: false })),
  ];

  if (!items.length) {
    container.innerHTML = "<p class='subtitle'>No active alerts for this area.</p>";
    return;
  }

  container.innerHTML = items
    .map(
      (alert) => `
      <article class="alert-card ${alert.isEarthquake ? "earthquake-card" : ""}">
        <h4>${alert.event || "Alert"}${alert.isEarthquake ? " 🫨" : ""}</h4>
        <p>${alert.headline || alert.area_desc || ""}</p>
        <p><strong>Severity:</strong> ${alert.severity || "Unknown"} · <strong>Time:</strong> ${alert.expires || "—"}</p>
      </article>`
    )
    .join("");
}

function renderPredictions(alerts) {
  const rows = [];
  alerts.forEach((alert) => {
    (alert.evacuation_predictions || []).forEach((prediction) => {
      const occ = prediction.inputs?.occupancy ?? "—";
      rows.push(`
        <tr>
          <td>${prediction.name}</td>
          <td>${alert.event || "Alert"}</td>
          <td>${occ}</td>
          <td>${(prediction.predicted_evacuation_rate * 100).toFixed(1)}%</td>
          <td>${prediction.predicted_evacuation_time_min}</td>
          <td><span class="${riskClass(prediction.risk_level)}">${prediction.risk_level}</span></td>
        </tr>
      `);
    });
  });

  document.getElementById("predictionsBody").innerHTML =
    rows.join("") || "<tr><td colspan='6'>No predictions yet.</td></tr>";
}

function renderMap(snapshot) {
  const center = snapshot.map_center || { lat: 38.58, lon: -121.3 };
  if (!map) {
    map = L.map("map").setView([center.lat, center.lon], 9);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap",
    }).addTo(map);
    markersLayer = L.layerGroup().addTo(map);
  }

  markersLayer.clearLayers();

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

  (snapshot.alerts || []).forEach((alert) => {
    if (alert.center_lat == null || alert.center_lon == null) return;
    L.circle([alert.center_lat, alert.center_lon], {
      color: "#ff6b6b",
      fillColor: "#ff6b6b",
      fillOpacity: 0.15,
      radius: 8000,
    })
      .bindPopup(`<strong>${alert.event}</strong><br/>${alert.headline || ""}`)
      .addTo(markersLayer);
  });

  (snapshot.earthquakes || []).forEach((quake) => {
    if (quake.center_lat == null || quake.center_lon == null) return;
    L.circleMarker([quake.center_lat, quake.center_lon], {
      radius: 12,
      color: "#ffd166",
      fillColor: "#ffd166",
      fillOpacity: 0.9,
    })
      .bindPopup(`<strong>🫨 ${quake.headline || quake.event}</strong>`)
      .addTo(markersLayer);
  });
}

function renderSnapshot(snapshot) {
  document.getElementById("alertCount").textContent =
    (snapshot.summary?.active_alerts ?? 0) + (snapshot.summary?.significant_earthquakes ?? 0);
  document.getElementById("riskCount").textContent = snapshot.summary?.high_risk_spots ?? 0;
  document.getElementById("spotCount").textContent = (snapshot.monitoring_spots || []).length;
  document.getElementById("modeBadge").textContent =
    snapshot.peoplesense_mode === "live" ? "PeopleSense: Live" : "PeopleSense: Placeholder";
  document.getElementById("lastUpdated").textContent = `Last updated: ${new Date(snapshot.generated_at).toLocaleString()}`;

  renderAlerts(snapshot.alerts || [], snapshot.earthquakes || []);
  renderPredictions(snapshot.alerts || []);
  renderMap(snapshot);
}

async function refresh() {
  const button = document.getElementById("refreshBtn");
  button.disabled = true;
  button.textContent = "Starting…";

  try {
    const sync = await fetch(apiUrl("/api/alerts/sync"), { method: "POST" });
    const info = await sync.json();

    if (info.status === "already_running") {
      startPipelinePolling(async () => {
        button.disabled = false;
        button.textContent = "Run Agent";
      });
      return;
    }

    if (!sync.ok) throw new Error("Pipeline trigger failed");

    button.textContent = "Pipeline running…";
    startPipelinePolling(async () => {
      try {
        const snapshot = await fetchDashboard();
        renderSnapshot(snapshot);
      } catch (error) {
        console.error(error);
      } finally {
        button.disabled = false;
        button.textContent = "Run Agent";
      }
    });
  } catch (error) {
    console.error(error);
    alert("Could not start the full emergency pipeline.");
    button.disabled = false;
    button.textContent = "Run Agent";
    showPipelineProgress(false);
  }
}

document.getElementById("refreshBtn").addEventListener("click", refresh);

export async function initApp() {
  try {
    const [snapshot, pipelineStatus] = await Promise.all([
      fetchDashboard(),
      fetchPipelineStatus(),
    ]);
    renderSnapshot(snapshot);

    if (pipelineStatus.status === "running") {
      const button = document.getElementById("refreshBtn");
      button.disabled = true;
      button.textContent = "Pipeline running…";
      startPipelinePolling(async () => {
        button.disabled = false;
        button.textContent = "Run Agent";
        const updated = await fetchDashboard();
        renderSnapshot(updated);
      });
    }
  } catch (error) {
    console.error(error);
    document.getElementById("alertsList").innerHTML =
      "<p class='subtitle'>Dashboard unavailable. Start the API server and refresh.</p>";
  }
}

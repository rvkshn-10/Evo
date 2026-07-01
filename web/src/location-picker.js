const API_BASE = import.meta.env.VITE_API_BASE || "";

let pickMode = false;
let blockHazardMode = false;
let pickMarker = null;
let routeLayer = null;
let hazardLayer = null;
let mapRef = null;
let lastAnalysis = null;
let lastPin = null;
let blockedHeadings = new Set();
let blockedPoints = [];

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function setPickMode(active) {
  pickMode = active;
  if (active) blockHazardMode = false;
  syncModeButtons();
}

function setBlockHazardMode(active) {
  blockHazardMode = active;
  if (active) pickMode = false;
  syncModeButtons();
}

function syncModeButtons() {
  const pickBtn = document.getElementById("pickLocationBtn");
  const blockBtn = document.getElementById("blockHazardBtn");
  const mapEl = document.getElementById("map");
  if (pickBtn) {
    pickBtn.classList.toggle("primary", pickMode);
    pickBtn.textContent = pickMode ? "Click map — pick site" : "Analyze location";
    pickBtn.setAttribute("aria-pressed", pickMode ? "true" : "false");
  }
  if (blockBtn) {
    blockBtn.classList.toggle("primary", blockHazardMode);
    blockBtn.setAttribute("aria-pressed", blockHazardMode ? "true" : "false");
  }
  if (mapEl) {
    mapEl.classList.toggle("map-pick-mode", pickMode);
    mapEl.classList.toggle("map-block-mode", blockHazardMode);
  }
}

function clearRouteLayer() {
  if (routeLayer && mapRef) {
    mapRef.removeLayer(routeLayer);
    routeLayer = null;
  }
}

function ensureHazardLayer() {
  if (!mapRef || !window.L) return null;
  if (!hazardLayer) {
    hazardLayer = window.L.layerGroup().addTo(mapRef);
  }
  return hazardLayer;
}

function redrawHazardMarkers() {
  const layer = ensureHazardLayer();
  if (!layer) return;
  layer.clearLayers();
  blockedPoints.forEach((point, index) => {
    window.L.circle([point.lat, point.lon], {
      radius: point.radius_m || 80,
      color: "#f87171",
      fillColor: "#f87171",
      fillOpacity: 0.25,
    })
      .bindPopup(`Blocked: ${point.reason || "hazard"}<br/><button type="button" data-remove-block="${index}">Remove</button>`)
      .addTo(layer);
    window.L.marker([point.lat, point.lon], {
      icon: window.L.divIcon({
        className: "hazard-block-icon",
        html: "🔥",
        iconSize: [24, 24],
      }),
    }).addTo(layer);
  });
}

function drawRoutes(analysis) {
  if (!mapRef || !window.L) return;
  clearRouteLayer();
  const loc = analysis.location || {};
  const routes = (analysis.evacuation_routes || []).filter((r) => !r.blocked);
  if (!loc.lat || !routes.length) return;

  routeLayer = window.L.layerGroup().addTo(mapRef);
  routes.forEach((route) => {
    const isBlocked = blockedHeadings.has(route.heading_deg);
    const color = isBlocked ? "#64748b" : route.recommended ? "#34d399" : "#94a3b8";
    const weight = route.recommended ? 4 : 2;
    window.L.polyline(
      [[loc.lat, loc.lon], [route.assembly_lat, route.assembly_lon]],
      {
        color,
        weight,
        dashArray: isBlocked || !route.recommended ? "6 8" : null,
        opacity: isBlocked ? 0.35 : 0.85,
      },
    )
      .bindPopup(
        `<strong>${route.compass}</strong>${route.is_detour ? " (detour)" : ""}<br/>`
        + `Walk: ${route.walk_distance_m} m · ~${route.estimated_clear_time_min} min clear`,
      )
      .addTo(routeLayer);
    if (!isBlocked) {
      window.L.circleMarker([route.assembly_lat, route.assembly_lon], {
        radius: route.recommended ? 7 : 5,
        color,
        fillColor: color,
        fillOpacity: 0.7,
      })
        .bindPopup(`${route.compass} assembly`)
        .addTo(routeLayer);
    }
  });
  redrawHazardMarkers();
}

function routeSourceLabel(source) {
  if (source === "evo1.4_route_head") return "Evo 1.4 route head";
  if (source === "haversine_fallback") return "Straight-line estimate";
  if (source === "osrm_foot") return "OSRM walking";
  return source || "";
}

function formatRouteList(routes) {
  const visible = (routes || []).filter((r) => !r.blocked);
  if (!visible.length) {
    return "<p class=\"subtitle\">No open routes — remove a blocker or hazard zone and recalculate.</p>";
  }
  return `<ul class="route-list">${visible
    .map((route) => {
      const heading = route.heading_deg;
      const isBlocked = blockedHeadings.has(heading);
      return `
      <li class="route-item ${route.recommended ? "route-recommended" : ""} ${isBlocked ? "route-blocked" : ""}">
        <div class="route-item-row">
          <strong>${route.rank}. ${route.compass}</strong>
          ${route.recommended ? "<span class=\"route-badge\">Recommended</span>" : ""}
          ${route.is_detour ? "<span class=\"route-badge detour\">Detour</span>" : ""}
          ${heading != null ? `<button type="button" class="btn neo-btn route-block-btn" data-block-heading="${heading}">${isBlocked ? "Unblock" : "Block exit"}</button>` : ""}
        </div>
        <span class="route-meta">
          ${route.walk_distance_m} m · ~${route.estimated_clear_time_min} min est. clear
          ${route.source ? ` · <span class="route-source">${routeSourceLabel(route.source)}</span>` : ""}
          ${route.heading_confidence != null ? ` · ${Math.round(route.heading_confidence * 100)}% conf.` : ""}
        </span>
      </li>`;
    })
    .join("")}</ul>`;
}

function getBlueprintPayload() {
  const url = document.getElementById("blueprintUrlInput")?.value?.trim();
  const exitCount = parseInt(document.getElementById("blueprintExitInput")?.value || "", 10);
  const notes = document.getElementById("blueprintNotesInput")?.value?.trim();
  if (!url && !exitCount && !notes) return null;
  return {
    url: url || null,
    exit_count: Number.isFinite(exitCount) ? exitCount : null,
    notes: notes || null,
  };
}

function renderAnalysisPanel(payload) {
  const panel = document.getElementById("locationAnalysisPanel");
  const body = document.getElementById("locationAnalysisBody");
  if (!panel || !body) return;

  const analysis = payload.analysis || payload;
  lastAnalysis = analysis;
  const loc = analysis.location || {};
  const ps = analysis.peoplesense || {};
  const pred = analysis.prediction || {};
  const best = analysis.recommended_route;
  const reason = document.getElementById("blockageReasonSelect")?.value || analysis.blockage_reason;

  body.innerHTML = `
    <div class="location-analysis-header">
      <h3>${loc.name || "Selected location"}</h3>
      <p class="subtitle">${loc.lat?.toFixed(5)}, ${loc.lon?.toFixed(5)} · ${loc.category || "—"}</p>
    </div>
    <div class="location-stats neo-inset">
      <div><span class="stat-label">PeopleSense</span><strong>${ps.occupancy ?? "—"}</strong> people</div>
      <div><span class="stat-label">Density</span><strong>${ps.density != null ? Number(ps.density).toFixed(2) : "—"}</strong></div>
      <div><span class="stat-label">Evac success</span><strong>${pred.predicted_evacuation_success_pct != null ? `${pred.predicted_evacuation_success_pct}%` : "—"}</strong></div>
      <div><span class="stat-label">Model time</span><strong>${pred.predicted_evacuation_time_min ?? "—"}</strong> min</div>
    </div>
    <div class="location-controls neo-inset">
      <label class="history-range-label">
        <span>Blockage reason</span>
        <select id="blockageReasonSelect" class="run-mode-select">
          <option value="">None</option>
          <option value="fire" ${reason === "fire" ? "selected" : ""}>Fire</option>
          <option value="flood" ${reason === "flood" ? "selected" : ""}>Flood</option>
          <option value="debris" ${reason === "debris" ? "selected" : ""}>Debris / collapse</option>
          <option value="police" ${reason === "police" ? "selected" : ""}>Police cordon</option>
        </select>
      </label>
      <label class="history-range-label">
        <span>Floor plan URL (optional)</span>
        <input id="blueprintUrlInput" class="run-mode-select" type="url" placeholder="https://… campus map PDF" value="${analysis.blueprint?.url || ""}" />
      </label>
      <label class="history-range-label">
        <span>Exits on plan (optional)</span>
        <input id="blueprintExitInput" class="run-mode-select" type="number" min="1" max="99" placeholder="6" value="${analysis.blueprint?.exit_count || ""}" />
      </label>
      <label class="history-range-label">
        <span>Plan notes</span>
        <input id="blueprintNotesInput" class="run-mode-select" type="text" placeholder="e.g. narrow west stairwell" value="${analysis.blueprint?.notes || ""}" />
      </label>
      <button id="recalcRoutesBtn" class="btn neo-btn primary" type="button">Recalculate detour routes</button>
    </div>
    ${best && !best.blocked ? `<p class="route-primary-hint">Best open egress: <strong>${best.compass}</strong>${best.is_detour ? " (detour)" : ""} · ~${best.estimated_clear_time_min} min${best.source ? ` · ${routeSourceLabel(best.source)}` : ""}</p>` : ""}
    ${blockedPoints.length ? `<p class="route-primary-hint">${blockedPoints.length} hazard zone(s) on map · ${blockedHeadings.size} blocked heading(s)</p>` : ""}
    <div id="routeListMount">${formatRouteList(analysis.evacuation_routes)}</div>
    <p class="sidebar-hint">${analysis.research_notice || ""}</p>
  `;

  panel.classList.remove("hidden");
  drawRoutes(analysis);

  document.getElementById("recalcRoutesBtn")?.addEventListener("click", () => {
    if (lastPin) analyzeAt(lastPin.lat, lastPin.lng);
  });
  body.querySelectorAll("[data-block-heading]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const heading = parseInt(btn.getAttribute("data-block-heading"), 10);
      if (blockedHeadings.has(heading)) blockedHeadings.delete(heading);
      else blockedHeadings.add(heading);
      if (lastPin) analyzeAt(lastPin.lat, lastPin.lng);
    });
  });
}

async function analyzeAt(lat, lng) {
  const panel = document.getElementById("locationAnalysisPanel");
  const body = document.getElementById("locationAnalysisBody");
  lastPin = { lat, lng };
  if (panel) panel.classList.remove("hidden");
  if (body) {
    body.innerHTML = "<p class=\"subtitle\">Analyzing — PeopleSense, blocked exits, Evo 1.3 routes…</p>";
  }

  const payload = {
    lat,
    lon: lng,
    use_evo13: true,
    use_llm: false,
    blocked_headings: [...blockedHeadings],
    blocked_points: blockedPoints,
    blockage_reason: document.getElementById("blockageReasonSelect")?.value || undefined,
    blueprint: getBlueprintPayload(),
  };

  try {
    const response = await fetch(apiUrl("/api/location/analyze"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderAnalysisPanel(await response.json());
  } catch (error) {
    console.error(error);
    if (body) {
      body.innerHTML = "<p class=\"subtitle\">Could not analyze this location. Is the API running?</p>";
    }
  } finally {
    setPickMode(false);
    setBlockHazardMode(false);
  }
}

function addHazardAt(lat, lng) {
  const reason = document.getElementById("blockageReasonSelect")?.value || "fire";
  blockedPoints.push({ lat, lon: lng, radius_m: 80, reason });
  redrawHazardMarkers();
  if (lastPin) analyzeAt(lastPin.lat, lastPin.lng);
  else analyzeAt(lat, lng);
}

function onMapClick(event) {
  if (!mapRef) return;
  const { lat, lng } = event.latlng;

  if (blockHazardMode) {
    addHazardAt(lat, lng);
    setBlockHazardMode(false);
    return;
  }

  if (!pickMode) return;

  if (pickMarker) {
    pickMarker.setLatLng([lat, lng]);
  } else if (window.L) {
    pickMarker = window.L.marker([lat, lng], { draggable: true }).addTo(mapRef);
    pickMarker.on("dragend", () => {
      const pos = pickMarker.getLatLng();
      blockedHeadings.clear();
      blockedPoints = [];
      analyzeAt(pos.lat, pos.lng);
    });
  }
  blockedHeadings.clear();
  blockedPoints = [];
  analyzeAt(lat, lng);
}

export function bindLocationPicker(map) {
  mapRef = map;
  if (!mapRef || mapRef._evoLocationPickerBound) return;
  mapRef._evoLocationPickerBound = true;
  mapRef.on("click", onMapClick);

  document.getElementById("pickLocationBtn")?.addEventListener("click", () => {
    setPickMode(!pickMode);
  });
  document.getElementById("blockHazardBtn")?.addEventListener("click", () => {
    setBlockHazardMode(!blockHazardMode);
  });
  document.getElementById("locationAnalysisClose")?.addEventListener("click", () => {
    document.getElementById("locationAnalysisPanel")?.classList.add("hidden");
    clearRouteLayer();
    if (hazardLayer) hazardLayer.clearLayers();
    setPickMode(false);
    setBlockHazardMode(false);
    blockedHeadings.clear();
    blockedPoints = [];
    lastPin = null;
  });
}

export function getLastLocationAnalysis() {
  return lastAnalysis;
}

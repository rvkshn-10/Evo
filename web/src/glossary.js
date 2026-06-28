/** Hover definitions for advanced metrics and model terms. */

export const GLOSSARY = {
  mae: {
    label: "MAE",
    text: "Mean Absolute Error — average prediction error in the same units as the target (percentage points for success, minutes for time). Lower is better.",
  },
  r2: {
    label: "R²",
    text: "Coefficient of determination — how much variance the model explains versus simply predicting the average. 1.0 is perfect; near 0 means little learned signal; negative means worse than the mean.",
  },
  r2_success: {
    label: "R² (success)",
    text: "Explained variance for evacuation success rate. Evo 1.2 hybrid keeps k-NN for success because this head hits a data ceiling on school/office sites.",
  },
  r2_time: {
    label: "R² (time)",
    text: "Explained variance for evacuation time in minutes. Reliable mainly for Train Station category (e.g. Cordova Park) in the current hybrid policy.",
  },
  mae_success: {
    label: "MAE (success)",
    text: "Average absolute error on evacuation success percentage across cross-validation folds.",
  },
  mae_time: {
    label: "MAE (time)",
    text: "Average absolute error on predicted evacuation time in minutes.",
  },
  quality_gates: {
    label: "Quality gates",
    text: "Fixed promotion checks (R², MAE, baseline beats, ONNX/OpenVINO parity, inference latency). All must pass before a model replaces production.",
  },
  hybrid: {
    label: "Hybrid inference",
    text: "Production policy: k-NN reference model for success and risk; Evo neural net for evacuation time at transit hubs only.",
  },
  dual_head: {
    label: "Dual-head",
    text: "Two prediction outputs from one pipeline: evacuation success (%) and evacuation time (minutes).",
  },
  mlp: {
    label: "MLP",
    text: "Multi-layer perceptron — feed-forward neural network head trained with grouped cross-validation.",
  },
  lightgbm: {
    label: "Gradient boosting",
    text: "LightGBM tree ensemble head; predictions are averaged with the MLP out-of-fold for robustness.",
  },
  openvino: {
    label: "Neural Compute Stick",
    text: "Intel USB accelerator (NCS1/NCS2). OpenVINO is the driver that routes the Evo model to the stick (MYRIAD) or CPU. Local Mac only — not on Vercel.",
  },
  accel_auto: {
    label: "Auto",
    text: "Detect a USB Neural Compute Stick (NCS2 preferred, then NCS1). Falls back to CPU if no stick is plugged in.",
  },
  accel_cpu: {
    label: "CPU",
    text: "Run Evo on the server CPU — no Neural Compute Stick required. Used on Vercel/Oracle production.",
  },
  accel_ncs2: {
    label: "Neural Compute Stick 2",
    text: "Intel NCS2 (Myriad X) over USB. Requires local API (python3 main.py) on the Mac with the stick plugged in.",
  },
  accel_ncs1: {
    label: "Neural Compute Stick 1",
    text: "Original Intel NCS1 (Myriad 2). Older hardware; may need legacy OpenVINO. Local Mac + USB only.",
  },
  onnx: {
    label: "ONNX Runtime",
    text: "Cross-platform CPU inference engine used when OpenVINO is unavailable or disabled.",
  },
  train_loss: {
    label: "Training loss",
    text: "Combined error while fitting the model on training folds. Should trend down but can overfit if validation diverges.",
  },
  val_loss: {
    label: "Validation loss",
    text: "Error on held-out folds during training — the check for generalization.",
  },
  data_ceiling: {
    label: "Data ceiling",
    text: "Model metrics plateau because labeled drill outcomes and timestamped occupancy are limited — more epochs cannot fix missing real data.",
  },
  hazard_features: {
    label: "Hazard features",
    text: "Live covariates from NOAA, USGS, and GDACS joined to each monitoring spot for context during prediction.",
  },
  run_mode_sync: {
    label: "Sync only (default)",
    text: "Pulls NOAA, USGS, FEMA, GDACS → PeopleSense occupancy → evacuation predictions (k-NN from reference data) → updates map and dashboard.",
    paid: "No",
    best: "Daily monitoring, free refreshes",
  },
  run_mode_evo: {
    label: "Evo 1.2 (hybrid)",
    text: "Same sync as Sync only, but predictions use the local Evo 1.2 hybrid policy (k-NN for success/risk; Evo neural net for evacuation time at transit hubs). Does not call Gemini, OpenAI, or any external LLM.",
    paid: "No — local ONNX/OpenVINO on the API server",
    best: "ML-backed predictions without API cost",
  },
  run_mode_external_ai: {
    label: "External AI",
    text: "Sync plus a short Gemini briefing (falls back to OpenAI mini, then GPT-4o if needed). Adds a plain-language situation summary on top of the dashboard.",
    paid: "Yes — API credits per run",
    best: "When you want a written briefing",
  },
  run_mode_broadcast: {
    label: "Full broadcast",
    text: "Sync plus the full 7-agent pipeline: researcher, writer, producer, script writer, charts, and files under output/ (sitrep, article, broadcast script, etc.).",
    paid: "Yes — many LLM calls per run",
    best: "Real emergency event → news package",
  },
  run_mode_evo13: {
    label: "Evo 1.3 (research)",
    text: "Research-only mode: live internet hazard feeds + enriched NIST/NFPA/transit reference data + k-NN estimates for all sites. Uses trained Evo 1.3 artifacts automatically when available. Not validated by FCUSD drills — may not be accurate for your schools.",
    paid: "No external LLM — local inference only",
    best: "Experiments and demos — not official FCUSD operations",
  },
  evo1_3_blocked: {
    label: "Why not Evo 1.3 yet?",
    text: "Evo 1.3 training is blocked until FCUSD provides measured drill outcomes (real_outcomes.json), drill-timestamp PeopleSense exports for Vista del Lago, Folsom High, and Cordova Park, and confirmed GPS. Without that labeled data, promotion gates fail (DATA_CEILING) and Evo 1.2 hybrid remains safer in production.",
    paid: "N/A",
    best: "Phase B after FCUSD evacuation drills",
  },
};

let tooltipEl = null;

function ensureTooltip() {
  if (tooltipEl) return tooltipEl;
  tooltipEl = document.createElement("div");
  tooltipEl.id = "glossaryTooltip";
  tooltipEl.className = "glossary-tooltip hidden";
  tooltipEl.setAttribute("role", "tooltip");
  document.body.appendChild(tooltipEl);
  return tooltipEl;
}

export function glossaryTerm(key, displayText) {
  const entry = GLOSSARY[key];
  const label = displayText ?? entry?.label ?? key;
  if (!entry) return label;
  return `<span class="glossary-term" data-glossary-key="${key}" tabindex="0">${label}</span>`;
}

export function initGlossaryTooltips(root = document) {
  const tip = ensureTooltip();

  const show = (term, event) => {
    const key = term.dataset.glossaryKey;
    const entry = GLOSSARY[key];
    if (!entry) return;
    let html = `<strong>${entry.label}</strong><p>${entry.text}</p>`;
    if (entry.paid) {
      html += `<p class="glossary-meta"><span>Uses paid AI?</span> ${entry.paid}</p>`;
    }
    if (entry.best) {
      html += `<p class="glossary-meta"><span>Best for</span> ${entry.best}</p>`;
    }
    tip.innerHTML = html;
    tip.classList.remove("hidden");
    positionTooltip(tip, event);
  };

  const hide = () => tip.classList.add("hidden");

  root.querySelectorAll(".glossary-term").forEach((term) => {
    term.addEventListener("mouseenter", (e) => show(term, e));
    term.addEventListener("focus", (e) => show(term, e));
    term.addEventListener("mousemove", (e) => positionTooltip(tip, e));
    term.addEventListener("mouseleave", hide);
    term.addEventListener("blur", hide);
  });
}

function positionTooltip(tip, event) {
  const margin = 12;
  const maxW = 300;
  let x = event.clientX + margin;
  let y = event.clientY + margin;
  const rect = tip.getBoundingClientRect();
  if (x + maxW > window.innerWidth - margin) {
    x = event.clientX - maxW - margin;
  }
  if (y + rect.height > window.innerHeight - margin) {
    y = event.clientY - rect.height - margin;
  }
  tip.style.left = `${Math.max(margin, x)}px`;
  tip.style.top = `${Math.max(margin, y)}px`;
}

import { initGlossaryTooltips } from "./glossary.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";

export const ACCELERATOR_LABELS = {
  auto: "Auto (detect USB stick)",
  cpu: "CPU only (no USB stick)",
  ncs2: "Neural Compute Stick 2",
  ncs1: "Neural Compute Stick 1",
};

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function useFixedMenu() {
  return window.matchMedia("(max-width: 960px)").matches;
}

export function isCloudHostedDashboard() {
  const host = window.location.hostname.toLowerCase();
  if (!host || host === "localhost" || host === "127.0.0.1") return false;
  return true;
}

export function initAcceleratorPicker({ onChange } = {}) {
  const select = document.getElementById("acceleratorSelect");
  const trigger = document.getElementById("acceleratorTrigger");
  const menu = document.getElementById("acceleratorMenu");
  const picker = document.querySelector(".accelerator-picker");
  if (!select || !trigger || !menu || !picker) return;

  const syncTriggerLabel = () => {
    trigger.textContent = ACCELERATOR_LABELS[select.value] || ACCELERATOR_LABELS.auto;
    menu.querySelectorAll(".run-mode-option").forEach((btn) => {
      const active = btn.dataset.accelerator === select.value;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
  };

  const resetMenuPosition = () => {
    menu.classList.remove("is-fixed");
    menu.style.top = "";
    menu.style.left = "";
    menu.style.width = "";
    menu.style.maxHeight = "";
  };

  const positionMenu = () => {
    if (!useFixedMenu()) {
      resetMenuPosition();
      return;
    }
    const rect = trigger.getBoundingClientRect();
    const margin = 8;
    const width = Math.min(rect.width, window.innerWidth - margin * 2);
    const left = Math.min(Math.max(margin, rect.left), window.innerWidth - width - margin);
    const spaceBelow = window.innerHeight - rect.bottom - margin;
    const spaceAbove = rect.top - margin;
    const openUp = spaceBelow < 180 && spaceAbove > spaceBelow;
    const maxHeight = Math.max(120, openUp ? spaceAbove - 8 : spaceBelow - 8);
    menu.classList.add("is-fixed");
    menu.style.width = `${width}px`;
    menu.style.left = `${left}px`;
    menu.style.maxHeight = `${maxHeight}px`;
    menu.style.top = openUp ? `${Math.max(margin, rect.top - maxHeight - 4)}px` : `${rect.bottom + 4}px`;
  };

  const closeMenu = () => {
    menu.classList.add("hidden");
    picker.classList.remove("is-open");
    trigger.setAttribute("aria-expanded", "false");
    resetMenuPosition();
  };

  const openMenu = () => {
    menu.classList.remove("hidden");
    picker.classList.add("is-open");
    trigger.setAttribute("aria-expanded", "true");
    positionMenu();
    initGlossaryTooltips(menu);
  };

  const applyAccelerator = async (value) => {
    if (!ACCELERATOR_LABELS[value]) return;
    if (isCloudHostedDashboard() && (value === "ncs1" || value === "ncs2")) {
      closeMenu();
      if (onChange) await onChange({ blocked_ncs_cloud: true });
      return;
    }
    select.value = value;
    syncTriggerLabel();
    closeMenu();

    const response = await fetch(apiUrl("/api/evo/accelerator"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ accelerator: value }),
    });
    const data = response.ok ? await response.json() : null;
    if (onChange) await onChange(data);
  };

  trigger.addEventListener("click", () => {
    if (menu.classList.contains("hidden")) openMenu();
    else closeMenu();
  });

  menu.querySelectorAll(".run-mode-option").forEach((btn) => {
    btn.addEventListener("click", () => applyAccelerator(btn.dataset.accelerator));
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".accelerator-picker")) closeMenu();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenu();
  });

  window.addEventListener("resize", () => {
    if (!menu.classList.contains("hidden")) positionMenu();
  });

  window.addEventListener(
    "scroll",
    () => {
      if (!menu.classList.contains("hidden") && useFixedMenu()) positionMenu();
    },
    true
  );

  syncTriggerLabel();
  initGlossaryTooltips(menu);
}

export function setAcceleratorDot(status) {
  const dot = document.getElementById("acceleratorStatusDot");
  if (!dot) return;
  dot.classList.remove("connected", "fallback", "warn");
  if (!status) return;

  const onStick =
    status.backend === "openvino" &&
    status.device &&
    String(status.device).toUpperCase().startsWith("MYRIAD");
  const requested = status.accelerator_requested || "auto";

  if (onStick) {
    dot.classList.add("connected");
    dot.title = status.status_message || "Neural Compute Stick active";
  } else if (status.loaded && status.backend === "onnxruntime") {
    dot.classList.add(requested === "ncs1" || requested === "ncs2" ? "warn" : "fallback");
    dot.title = status.status_message || "Running on CPU (no stick detected)";
  } else {
    dot.title = status.status_message || "Checking device…";
  }
}

export function syncAcceleratorSelectFromStatus(status) {
  const select = document.getElementById("acceleratorSelect");
  const trigger = document.getElementById("acceleratorTrigger");
  if (!select || !status?.accelerator_requested) return;
  select.value = status.accelerator_requested;
  if (trigger) {
    trigger.textContent =
      ACCELERATOR_LABELS[status.accelerator_requested] || ACCELERATOR_LABELS.auto;
  }
}

import { initGlossaryTooltips } from "./glossary.js";

export const RUN_MODE_LABELS = {
  sync: "Sync only (default)",
  external_ai: "External AI (Gemini → OpenAI)",
  evo: "Evo 1.2 hybrid (production)",
  evo13: "Evo 1.3 (research)",
  broadcast: "Full broadcast",
};

function useFixedMenu() {
  return window.matchMedia("(max-width: 960px)").matches;
}

export function initRunModePicker() {
  const select = document.getElementById("runModeSelect");
  const trigger = document.getElementById("runModeTrigger");
  const menu = document.getElementById("runModeMenu");
  const picker = document.querySelector(".run-mode-picker");
  if (!select || !trigger || !menu || !picker) return;

  const syncTriggerLabel = () => {
    trigger.textContent = RUN_MODE_LABELS[select.value] || RUN_MODE_LABELS.sync;
    menu.querySelectorAll(".run-mode-option").forEach((btn) => {
      const active = btn.dataset.mode === select.value;
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
    const left = Math.min(
      Math.max(margin, rect.left),
      window.innerWidth - width - margin
    );
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

  const setMode = (mode) => {
    if (!RUN_MODE_LABELS[mode]) return;
    select.value = mode;
    syncTriggerLabel();
    closeMenu();
    select.dispatchEvent(new Event("change", { bubbles: true }));
  };

  trigger.addEventListener("click", () => {
    if (menu.classList.contains("hidden")) openMenu();
    else closeMenu();
  });

  menu.querySelectorAll(".run-mode-option").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".run-mode-picker")) closeMenu();
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

  select.addEventListener("change", syncTriggerLabel);
  syncTriggerLabel();
  initGlossaryTooltips(menu);
}

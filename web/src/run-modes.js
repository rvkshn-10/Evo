import { initGlossaryTooltips } from "./glossary.js";

export const RUN_MODE_LABELS = {
  sync: "Sync only (default)",
  external_ai: "External AI (Gemini → OpenAI)",
  evo: "Evo 1.2 hybrid (production)",
  evo13: "Evo 1.3 (research)",
  broadcast: "Full broadcast",
};

export function initRunModePicker() {
  const select = document.getElementById("runModeSelect");
  const trigger = document.getElementById("runModeTrigger");
  const menu = document.getElementById("runModeMenu");
  if (!select || !trigger || !menu) return;

  const syncTriggerLabel = () => {
    trigger.textContent = RUN_MODE_LABELS[select.value] || RUN_MODE_LABELS.sync;
    menu.querySelectorAll(".run-mode-option").forEach((btn) => {
      const active = btn.dataset.mode === select.value;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
  };

  const closeMenu = () => {
    menu.classList.add("hidden");
    trigger.setAttribute("aria-expanded", "false");
  };

  const openMenu = () => {
    menu.classList.remove("hidden");
    trigger.setAttribute("aria-expanded", "true");
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

  select.addEventListener("change", syncTriggerLabel);
  syncTriggerLabel();
  initGlossaryTooltips(menu);
}

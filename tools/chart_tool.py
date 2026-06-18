from langchain.tools import tool
import logging
import os
import json

logger = logging.getLogger(__name__)


@tool
def generate_population_chart(event_folder: str) -> str:
    """
    Read population_log.json from the event folder and generate a time-series
    line chart showing affected population across all event notifications.
    Saves the chart as population_chart.png in the event folder.

    Args:
        event_folder: Path to the event output folder

    Returns:
        Path to the saved chart file, or an error message
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    log_path = os.path.join(event_folder, "population_log.json")
    if not os.path.exists(log_path):
        return f"Error: population_log.json not found at {log_path}"

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        return f"Error reading population_log.json: {e}"

    if not entries:
        return "Error: population_log.json is empty"

    # Parse timestamps — try ISO-8601, fall back to index
    timestamps = []
    populations = []
    labels = []
    for entry in entries:
        ts_raw = entry.get("timestamp", "")
        pop = entry.get("affected_population", 0)
        status = entry.get("status", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            ts = None
        timestamps.append(ts)
        populations.append(pop)
        labels.append(status.upper())

    use_dates = all(t is not None for t in timestamps)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    x = timestamps if use_dates else list(range(len(entries)))

    ax.plot(x, populations, color="#e94560", linewidth=2.5, marker="o",
            markersize=8, markerfacecolor="#e94560", markeredgecolor="white",
            markeredgewidth=1.5, zorder=3)

    # Annotate each point with status label and population
    for i, (xi, pop, lbl) in enumerate(zip(x, populations, labels)):
        ax.annotate(
            f"{lbl}\n{pop:,}",
            xy=(xi, pop),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460", edgecolor="#e94560", alpha=0.85),
        )

    # Fill under the line
    ax.fill_between(x, populations, alpha=0.15, color="#e94560")

    ax.set_title("Affected Population Over Time", color="white", fontsize=14, pad=15)
    ax.set_xlabel("Event Timeline", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Estimated Affected Population", color="#aaaaaa", fontsize=10)

    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f3460")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    if use_dates:
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

    ax.grid(True, color="#0f3460", linestyle="--", alpha=0.5)

    chart_path = os.path.join(event_folder, "population_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info(f"[generate_population_chart] Saved chart to {chart_path}")
    return f"Chart saved: {chart_path}"

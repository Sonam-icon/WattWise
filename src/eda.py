
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_cycles(
    df: pd.DataFrame,
    power_col: str,
    threshold_w: float = 5.0,
    min_duration_minutes: int = 5,
    merge_gap_minutes: int = 3
) -> pd.DataFrame:

    """Detect washing-machine cycles without bridging missing-data gaps."""

    d = (
        df[["timestamp", power_col]]
        .copy()
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )

    s = pd.to_numeric(d[power_col], errors="coerce").clip(lower=0)

    if s.empty:
        return pd.DataFrame()

    active = s >= threshold_w
    known = s.notna()

    # Identify contiguous known-data segments.
    # Missing readings break continuity.
    missing_groups = (~known).cumsum()

    events = []

    for _, segment in active[known].groupby(missing_groups[known]):

        segment = segment.copy()

        # Merge short OFF gaps only within a continuous known segment.
        if merge_gap_minutes > 0:

            off = ~segment
            groups = (off != off.shift(fill_value=False)).cumsum()

            for _, idx in off[off].groupby(groups[off]).groups.items():

                if len(idx) <= merge_gap_minutes:

                    first_pos = segment.index.get_loc(idx[0])
                    last_pos = segment.index.get_loc(idx[-1])

                    # Merge only if OFF gap is surrounded by active readings.
                    if first_pos > 0 and last_pos < len(segment) - 1:
                        before = segment.iloc[first_pos - 1]
                        after = segment.iloc[last_pos + 1]

                        if before and after:
                            segment.loc[idx] = True

        # Identify active runs after gap merging.
        runs = (segment != segment.shift(fill_value=False)).cumsum()

        for _, cycle in segment.groupby(runs):

            if not bool(cycle.iloc[0]):
                continue

            start = cycle.index[0]
            end = cycle.index[-1]

            duration = (
                end - start
            ).total_seconds() / 60.0 + 1.0

            if duration < min_duration_minutes:
                continue

            values = s.loc[start:end].dropna()

            if values.empty:
                continue

            energy = float(values.sum() / 60000.0)

            events.append({
                "start": start,
                "end": end,
                "duration_min": float(duration),
                "peak_power_w": float(values.max()),
                "energy_kwh": energy
            })

    return pd.DataFrame(
        events,
        columns=[
            "start",
            "end",
            "duration_min",
            "peak_power_w",
            "energy_kwh"
        ]
    )


def household_summary(
    df: pd.DataFrame,
    power_col: str,
    house: int,
    threshold_w: float = 5.0
) -> dict:

    s = pd.to_numeric(df[power_col], errors="coerce")

    cycles = detect_cycles(
        df,
        power_col,
        threshold_w=threshold_w
    )

    valid = s.dropna()

    return {
        "house": house,
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "rows": len(df),
        "missing_pct": float(s.isna().mean() * 100),
        "off_pct": (
            float((valid < threshold_w).mean() * 100)
            if len(valid) else np.nan
        ),
        "cycles": len(cycles),
        "median_cycle_duration_min": (
            float(cycles["duration_min"].median())
            if not cycles.empty else np.nan
        ),
        "median_cycle_energy_kwh": (
            float(cycles["energy_kwh"].median())
            if not cycles.empty else np.nan
        )
    }


def make_eda_plots(
    df: pd.DataFrame,
    aggregate_col: str,
    wm_col: str,
    output_dir: str | Path,
    house: int
) -> list[str]:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []

    d = (
        df.dropna(subset=["timestamp"])
        .copy()
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )

    # Representative day with sufficient observations AND active washing-machine use
    daily_counts = d.groupby(d.index.date)[wm_col].count()
    daily_max = d.groupby(d.index.date)[wm_col].max()
    daily_sum = d.groupby(d.index.date)[wm_col].sum()

    valid_days = daily_counts[daily_counts >= 20 * 60]

    # Select a day that has both sufficient data AND a prominent active washing cycle
    active_days = valid_days.index[valid_days.index.isin(daily_max[daily_max >= 500.0].index)]

    if len(active_days) > 0:
        # Choose the active day with the highest total energy to clearly show the signature
        selected_day = daily_sum.loc[active_days].idxmax()
    elif not valid_days.empty:
        selected_day = valid_days.index[0]
    else:
        selected_day = None

    if selected_day is not None:
        day = d.loc[d.index.date == selected_day]

        fig, ax = plt.subplots(figsize=(12, 4.5))

        ax.plot(day.index, day[aggregate_col], label="Aggregate power", color="#1f77b4", linewidth=1.0, alpha=0.8)
        ax.plot(day.index, day[wm_col], label="Washing machine", color="#d62728", linewidth=1.2)

        ax.set_title(f"House {house}: Representative 24-Hour Trace ({selected_day})")
        ax.set_ylabel("Active Power (Watts)")
        ax.set_xlabel("Time of Day")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        fig.tight_layout()

        p = out / f"house_{house}_representative_day.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)

        paths.append(str(p))

    # Hourly profile
    d["hour"] = d.index.hour

    hourly = d.groupby("hour")[wm_col].mean()

    fig, ax = plt.subplots(figsize=(8, 4))

    hourly.plot(kind="bar", ax=ax)

    ax.set_title(f"House {house}: mean washing-machine power by hour")
    ax.set_ylabel("Power (W)")

    fig.tight_layout()

    p = out / f"house_{house}_hourly_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    paths.append(str(p))

    # Weekday profile
    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    d["day_of_week"] = d.index.day_name()

    daily = d.groupby("day_of_week")[wm_col].mean().reindex(
        weekday_order
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    daily.plot(kind="bar", ax=ax)

    ax.set_title(f"House {house}: mean washing-machine power by weekday")
    ax.set_ylabel("Power (W)")

    fig.tight_layout()

    p = out / f"house_{house}_weekday_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    paths.append(str(p))

    # Weekday versus weekend
    d["day_type"] = np.where(
        d.index.dayofweek >= 5,
        "Weekend",
        "Weekday"
    )

    day_type = d.groupby("day_type")[wm_col].mean()

    fig, ax = plt.subplots(figsize=(6, 4))

    day_type.plot(kind="bar", ax=ax)

    ax.set_title(f"House {house}: weekday vs weekend")
    ax.set_ylabel("Mean power (W)")

    fig.tight_layout()

    p = out / f"house_{house}_weekend_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    paths.append(str(p))

    # Detected washing-machine cycles (plotted on relative elapsed-time axis)
    cycles = detect_cycles(d.reset_index(), wm_col)

    if not cycles.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # Select up to 3 distinct prominent cycles (sorted by energy or distinct events)
        sample_cycles = cycles.sort_values("energy_kwh", ascending=False).head(3)

        for i, (_, event) in enumerate(sample_cycles.iterrows(), start=1):
            seg = d.loc[event["start"]:event["end"], wm_col]
            elapsed_min = (seg.index - event["start"]).total_seconds() / 60.0

            label_str = (
                f"Cycle {i}: {event['start'].strftime('%Y-%m-%d %H:%M')} "
                f"({event['duration_min']:.0f}m, {event['energy_kwh']:.2f} kWh, peak {event['peak_power_w']:.0f}W)"
            )
            ax.plot(elapsed_min, seg.values, linewidth=1.5, label=label_str)

        ax.set_title(f"House {house}: Detected Washing-Machine Cycles (Relative Time Overlay)")
        ax.set_xlabel("Elapsed Time from Cycle Start (Minutes)")
        ax.set_ylabel("Power (Watts)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9)

        fig.tight_layout()

        p = out / f"house_{house}_cycles.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)

        paths.append(str(p))

    return paths


def plot_cross_house_comparison(
    summaries_df: pd.DataFrame,
    output_dir: str | Path
) -> str:
    """Generate cross-household comparison charts for washing machine cycles."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    houses = [f"House {h}" for h in summaries_df["house"]]

    # Panel 1: Median Cycle Duration
    axes[0].bar(houses, summaries_df["median_cycle_duration_min"], color="#2ca02c", width=0.55)
    axes[0].set_title("Median Cycle Duration (Minutes)")
    axes[0].set_ylabel("Minutes")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(summaries_df["median_cycle_duration_min"]):
        axes[0].text(i, v + 1.5, f"{v:.0f}m", ha="center", fontweight="bold")

    # Panel 2: Median Energy per Cycle
    axes[1].bar(houses, summaries_df["median_cycle_energy_kwh"], color="#1f77b4", width=0.55)
    axes[1].set_title("Median Energy per Cycle (kWh)")
    axes[1].set_ylabel("kWh")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(summaries_df["median_cycle_energy_kwh"]):
        axes[1].text(i, v + 0.015, f"{v:.2f}", ha="center", fontweight="bold")

    # Panel 3: Total Detected Cycles
    axes[2].bar(houses, summaries_df["cycles"], color="#ff7f0e", width=0.55)
    axes[2].set_title("Total Detected Cycles")
    axes[2].set_ylabel("Number of Cycles")
    axes[2].grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(summaries_df["cycles"]):
        axes[2].text(i, v + 15, f"{v}", ha="center", fontweight="bold")

    fig.suptitle("Cross-Household Washing Machine Cycle Comparison (REFIT)", fontsize=13, y=1.03)
    fig.tight_layout()

    out_path = out / "cross_house_cycles_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)



"""Run raw data cleaning, washing-machine EDA and disaggregation."""

from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.io_utils import (
    find_household_csv,
    read_household_data,
    mask_invalid_appliance_readings,
)
from src.clean import clean_household_csv
from src.eda import (
    household_summary,
    make_eda_plots,
    plot_cross_house_comparison,
)
from src.features import build_features
from src.model import train_evaluate


# ==================================================
# PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


# ==================================================
# RESAMPLING FUNCTION
# ==================================================

def resample_cleaned_to_1min(df):
    """
    Convert cleaned REFIT readings to one-minute average power.

    Power readings are averaged within each minute.
    Missing minutes are retained as NaN rather than filled with zero.
    """

    d = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .copy()
        .set_index("timestamp")
    )

    power_cols = [
        col for col in d.columns
        if col == "Aggregate" or col.startswith("Appliance")
    ]

    if not power_cols:
        raise ValueError(
            "No aggregate or appliance power columns found."
        )

    result = d[power_cols].resample("1min").mean()

    return result.reset_index()


# ==================================================
# MAIN PIPELINE
# ==================================================

def main():

    # Create output directories
    cleaning_dir = OUT / "cleaning"
    eda_dir = OUT / "eda"
    model_dir = OUT / "model"

    for folder in [cleaning_dir, eda_dir, model_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # TASK 1: RAW HOUSE 1 CLEANING
    # ==================================================

    print("\n========== TASK 1: RAW DATA CLEANING ==========")

    raw_paths = find_household_csv(DATA / "Raw", 1)

    print("Raw House 1 files found:")
    for path in raw_paths:
        print(" -", path.name)

    # Combine raw House 1 parts
    raw = read_household_data(raw_paths)

    print("Combined raw data shape:", raw.shape)

    # Clean and resample to one-minute intervals
    clean, report = clean_household_csv(
        raw,
        cleaning_dir / "house_1_cleaned_1min.csv"
    )

    (cleaning_dir / "house_1_quality_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8"
    )

    print("Raw cleaning completed.")
    print("Cleaned data shape:", clean.shape)

    # Before-and-after plots
    selected_columns = [
        "Aggregate",
        "Appliance1",
        "Appliance2",
    ]

    for col in selected_columns:
        if col not in raw.columns or col not in clean.columns:
            raise KeyError(
                f"Required plotting column '{col}' not found."
            )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(15, 10)
    )

    for i, col in enumerate(selected_columns):

        axes[i, 0].plot(
            raw["timestamp"],
            raw[col],
            linewidth=0.5
        )
        axes[i, 0].set_title(f"Raw data: {col}")
        axes[i, 0].set_ylabel("Power (W)")

        axes[i, 1].plot(
            clean["timestamp"],
            clean[col],
            linewidth=0.8
        )
        axes[i, 1].set_title(f"Cleaned 1-minute data: {col}")
        axes[i, 1].set_ylabel("Power (W)")

    fig.tight_layout()

    fig.savefig(
        cleaning_dir / "house_1_before_after.png",
        dpi=150
    )

    plt.close(fig)

    print("Before/after plots saved.")

    # ==================================================
    # TASK 2 AND TASK 3: EDA + MODEL
    # ==================================================

    print("\n========== TASK 2 AND 3: EDA + MODEL ==========")

    mapping_path = ROOT / "config" / "household_channels.json"

    if not mapping_path.exists():
        raise FileNotFoundError(
            "Missing config/household_channels.json."
        )

    mapping = json.loads(
        mapping_path.read_text(encoding="utf-8")
    )

    summaries = []
    data_quality = []

    for house_str, cfg in mapping.items():

        house = int(house_str)

        print(f"\nProcessing House {house}...")

        paths = find_household_csv(DATA / "Cleaned", house)
        df_original = read_household_data(paths)

        # Validate required columns before quality filtering
        aggregate_col = cfg["aggregate"]
        washing_machine_col = cfg["washing_machine"]

        for col in [
            aggregate_col,
            washing_machine_col,
            "Issues",
        ]:
            if col not in df_original.columns:
                raise KeyError(
                    f"House {house}: required column '{col}' "
                    "not found in original data."
                )

        # Count original quality flags before masking
        issue_mask = (
            pd.to_numeric(
                df_original["Issues"],
                errors="coerce"
            ).eq(1)
        )

        issue_count = int(issue_mask.sum())

        issue_pct = (
            issue_count / len(df_original) * 100
            if len(df_original) > 0
            else 0
        )

        # Conservatively mask all appliance readings
        # where REFIT marks an aggregate/submeter inconsistency.
        # Aggregate readings are preserved.
        df_valid = mask_invalid_appliance_readings(
            df_original
        )

        # Resample AFTER masking so invalid readings
        # cannot influence minute-level appliance averages.
        df = resample_cleaned_to_1min(df_valid)

        # Validate required columns after resampling
        for col in [aggregate_col, washing_machine_col]:
            if col not in df.columns:
                raise KeyError(
                    f"House {house}: column '{col}' "
                    "not found after resampling."
                )

        print("Original data shape:", df_original.shape)
        print("One-minute data shape:", df.shape)
        print("Washing-machine channel:", washing_machine_col)
        print(
            "Original Issues == 1 rows:",
            issue_count,
            f"({issue_pct:.2f}%)"
        )
        print(
            "Date range:",
            df["timestamp"].min(),
            "to",
            df["timestamp"].max()
        )

        # Basic data quality information
        data_quality.append({
            "house": house,
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
            "rows_1min": int(len(df)),
            "original_issue_rows": issue_count,
            "original_issue_pct": round(issue_pct, 2),
            "missing_aggregate": int(
                df[aggregate_col].isna().sum()
            ),
            "missing_washing_machine": int(
                df[washing_machine_col].isna().sum()
            ),
            "washing_machine_missing_pct": round(
                df[washing_machine_col].isna().mean() * 100,
                2
            ),
        })

        # Household summary and cycle statistics
        summary = household_summary(
            df,
            washing_machine_col,
            house
        )

        summaries.append(summary)

        # EDA plots
        make_eda_plots(
            df,
            aggregate_col,
            washing_machine_col,
            eda_dir,
            house
        )

        # Train model for selected household
        if house == int(cfg.get("model_house", 1)):

            print(
                f"Training disaggregation model for House {house}..."
            )

            X, feature_info = build_features(
                df,
                aggregate_col
            )

            predictions, model_results = train_evaluate(
                df,
                X,
                washing_machine_col,
                model_dir
            )

            # Save model metrics
            (model_dir / "metrics.json").write_text(
                json.dumps(
                    model_results,
                    indent=2,
                    default=str
                ),
                encoding="utf-8"
            )

            # Save feature names
            (model_dir / "features.json").write_text(
                json.dumps(
                    feature_info,
                    indent=2
                ),
                encoding="utf-8"
            )

            print("Model evaluation completed.")
            print(
                "Selected model:",
                model_results["selected_model"]
            )

    # ==================================================
    # SAVE FINAL SUMMARIES
    # ==================================================

    summaries_df = pd.DataFrame(summaries)

    summaries_df.to_csv(
        eda_dir / "household_summary.csv",
        index=False
    )

    if not summaries_df.empty:
        cross_plot_path = plot_cross_house_comparison(
            summaries_df,
            eda_dir
        )

        print(
            "Cross-household comparison plot saved:",
            cross_plot_path
        )

    pd.DataFrame(data_quality).to_csv(
        eda_dir / "household_data_quality.csv",
        index=False
    )

    print("\n========== PIPELINE COMPLETED ==========")
    print(f"Outputs saved under: {OUT}")


if __name__ == "__main__":
    main()

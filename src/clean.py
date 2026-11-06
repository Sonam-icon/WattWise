
from pathlib import Path

import pandas as pd


def inspect_raw(
    df: pd.DataFrame,
    interval_tolerance_seconds: float = 2.0,
    extreme_threshold_w: float = 4000.0
) -> dict:
    """
    Inspect raw REFIT data before cleaning.

    Small timestamp variations within the configured tolerance
    are not treated as irregular intervals.

    Long gaps are reported separately.
    Extreme positive readings are flagged, not automatically removed.
    """

    ts = df["timestamp"].dropna().sort_values()

    # Use unique timestamps to avoid duplicate records distorting
    # the sampling interval calculation.
    unique_ts = ts.drop_duplicates()

    diffs = unique_ts.diff().dropna()
    interval_seconds = diffs.dt.total_seconds()

    median_interval = (
        float(interval_seconds.median())
        if len(interval_seconds) else None
    )

    numeric = df.select_dtypes(include="number")

    # Unix is a timestamp-related field, not a power channel.
    power_cols = [
        c for c in numeric.columns
        if c.lower() != "unix"
    ]

    if median_interval is not None:
        irregular_intervals = int(
            (
                (interval_seconds - median_interval).abs()
                > interval_tolerance_seconds
            ).sum()
        )

        # Flag intervals exceeding 1.5 times the median interval.
        long_gap_intervals = int(
            (interval_seconds > median_interval * 1.5).sum()
        )
    else:
        irregular_intervals = 0
        long_gap_intervals = 0

    return {
        "rows": int(len(df)),
        "start": str(ts.min()) if len(ts) else None,
        "end": str(ts.max()) if len(ts) else None,

        "invalid_timestamp_rows": int(
            df.attrs.get("invalid_timestamp_rows", 0)
        ),

        "duplicate_timestamps": int(
            df["timestamp"].duplicated().sum()
        ),

        "median_interval_seconds": median_interval,
        "interval_tolerance_seconds": interval_tolerance_seconds,
        "irregular_intervals": irregular_intervals,
        "long_gap_intervals": long_gap_intervals,

        "negative_readings_by_column": {
            c: int((numeric[c] < 0).sum())
            for c in power_cols
        },

        "maximum_reading_by_column": {
            c: (
                float(numeric[c].max())
                if numeric[c].notna().any()
                else None
            )
            for c in power_cols
        },

        "readings_above_extreme_threshold": {
            c: int((numeric[c] > extreme_threshold_w).sum())
            for c in power_cols
        },

        "extreme_reading_threshold_w": extreme_threshold_w,

        "missing_values_by_column": {
            c: int(df[c].isna().sum())
            for c in df.columns
        }
    }


def count_missing_runs(series: pd.Series) -> int:
    """Count separate consecutive runs of missing values."""

    missing = series.isna()

    if not missing.any():
        return 0

    return int(
        (missing & ~missing.shift(1, fill_value=False)).sum()
    )


def interpolate_short_internal_gaps(
    df: pd.DataFrame,
    max_gap_minutes: int = 3
) -> pd.DataFrame:
    """
    Interpolate only complete internal missing runs of up to
    max_gap_minutes.

    Longer gaps and boundary gaps remain missing.
    """

    result = df.copy()

    for col in result.columns:
        series = result[col].copy()
        missing = series.isna().to_numpy()

        i = 0

        while i < len(series):

            if not missing[i]:
                i += 1
                continue

            # Start of a consecutive missing run.
            start = i

            while i < len(series) and missing[i]:
                i += 1

            end = i - 1
            gap_length = end - start + 1

            # Interpolate only if:
            # 1. The complete gap is within the allowed length.
            # 2. A valid observation exists before the gap.
            # 3. A valid observation exists after the gap.
            if (
                gap_length <= max_gap_minutes
                and start > 0
                and i < len(series)
            ):
                local = series.iloc[start - 1:i + 1]

                interpolated = local.interpolate(
                    method="time"
                )

                result.iloc[
                    start:i,
                    result.columns.get_loc(col)
                ] = interpolated.iloc[1:-1].to_numpy()

    return result


def clean_household_csv(
    raw: pd.DataFrame,
    output_path: str | Path,
    frequency: str = "1min",
    short_gap_minutes: int = 3
) -> tuple[pd.DataFrame, dict]:
    """
    Clean raw REFIT household readings.

    Steps:
    1. Inspect original data.
    2. Sort timestamps.
    3. Remove duplicate timestamps, keeping the last record.
    4. Mask negative power readings.
    5. Resample power channels to one-minute mean values.
    6. Interpolate only complete short internal gaps.
    7. Retain longer gaps and boundary gaps as missing values.
    8. Save cleaned data and quality report.
    """

    # Inspect original data before modifying it.
    report = inspect_raw(raw)

    # Sort chronologically and remove duplicate timestamps.
    clean = (
        raw.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .copy()
    )

    report["rows_after_duplicate_removal"] = int(len(clean))
    report["duplicates_removed"] = (
        report["rows"] - len(clean)
    )

    clean = clean.set_index("timestamp")

    # Select numeric power columns, excluding Unix.
    numeric_cols = [
        c for c in clean.select_dtypes(include="number").columns
        if c.lower() != "unix"
    ]

    # Negative power readings are invalid.
    clean[numeric_cols] = clean[numeric_cols].mask(
        clean[numeric_cols] < 0
    )

    # Resample power readings to one-minute mean values.
    resampled = clean[numeric_cols].resample(frequency).mean()

    missing_before_interpolation = (
        resampled.isna().sum().astype(int).to_dict()
    )

    # Interpolate only complete short internal gaps.
    if short_gap_minutes > 0:
        resampled = interpolate_short_internal_gaps(
            resampled,
            max_gap_minutes=short_gap_minutes
        )

    # Missing readings remaining after interpolation.
    missing_after = (
        resampled.isna().sum().astype(int).to_dict()
    )

    # Count consecutive missing runs separately for each channel.
    missing_runs_by_column = {
        c: count_missing_runs(resampled[c])
        for c in numeric_cols
    }

    # Count timestamps where at least one power channel is missing.
    timestamps_with_remaining_missing = int(
        resampled.isna().any(axis=1).sum()
    )

    # Count timestamps where all power channels are missing.
    timestamps_all_channels_missing = int(
        resampled.isna().all(axis=1).sum()
    )

    result = resampled.reset_index()

    # Save cleaned CSV.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)

    # Update quality report.
    report.update({
        "rows_after_resampling": int(len(result)),

        "power_columns_resampled": numeric_cols,

        "missing_before_interpolation": (
            missing_before_interpolation
        ),

        "missing_after_short_gap_fill": missing_after,

        "remaining_missing_runs_by_column": (
            missing_runs_by_column
        ),

        "timestamps_with_remaining_missing_values": (
            timestamps_with_remaining_missing
        ),

        "timestamps_all_power_channels_missing": (
            timestamps_all_channels_missing
        ),

        "duplicate_rule": (
            "Sort by timestamp and keep the last duplicate record"
        ),

        "negative_reading_rule": (
            "Negative power readings are replaced with missing values"
        ),

        "extreme_reading_rule": (
            "Readings above the configured threshold are flagged "
            "for investigation, not automatically removed"
        ),

        "extreme_readings_investigation_outputs": {
            "analysis_csv": (
                "outputs/cleaning/"
                "house_1_extreme_readings_analysis.csv"
            ),
            "summary_json": (
                "outputs/cleaning/"
                "house_1_extreme_readings_summary.json"
            )
        },

        "resampling": (
            f"{frequency} mean of available power readings"
        ),

        "gap_rule": (
            f"Interpolate only complete internal gaps of up to "
            f"{short_gap_minutes} consecutive missing minutes; "
            "longer gaps and boundary gaps remain missing"
        ),

        "unix_column_rule": (
            "Unix excluded from power resampling because it is "
            "a timestamp-related field, not a power measurement"
        )
    })

    return result, report


from pathlib import Path
import pandas as pd


def find_household_csv(folder, house):
    """Find the relevant REFIT CSV file(s) for a household."""
    folder = Path(folder)

    # Raw data can be split into multiple parts.
    raw_matches = sorted(
        folder.glob(f"RAW_House{house}_Part*.csv")
    )

    if raw_matches:
        return raw_matches

    # Cleaned data is generally one file per household.
    patterns = [
        f"CLEAN_House{house}.csv",
        f"House{house}.csv",
        f"House_{house}.csv"
    ]

    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches

    raise FileNotFoundError(
        f"Could not find House {house} CSV under {folder}"
    )


def read_refit_csv(path):
    """Read one REFIT CSV and standardize its timestamp column."""
    df = pd.read_csv(path, low_memory=False)

    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Unexpected/empty REFIT file: {path}")

    timestamp_col = (
        "Time" if "Time" in df.columns else df.columns[0]
    )

    ts = pd.to_datetime(
    df[timestamp_col],
    errors="coerce",
    format="%Y-%m-%d %H:%M:%S"
)

    invalid_timestamp_count = int(ts.isna().sum())

    df = df.drop(columns=[timestamp_col])
    df.insert(0, "timestamp", ts)

    df = df.dropna(subset=["timestamp"]).copy()

    # Convert sensor readings to numeric values.
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Preserve invalid timestamp count for quality reporting.
    df.attrs["invalid_timestamp_rows"] = invalid_timestamp_count

    return df


def read_household_data(paths):
    """Read and combine one or multiple CSV files for a household."""

    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = [
        read_refit_csv(path)
        for path in paths
    ]

    # Add invalid timestamp counts across all source files.
    invalid_timestamp_count = sum(
        frame.attrs.get("invalid_timestamp_rows", 0)
        for frame in frames
    )

    df = pd.concat(frames, ignore_index=True)

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    df.attrs["invalid_timestamp_rows"] = invalid_timestamp_count

    return df


def resolve_column(df, requested):
    """Resolve a column name without case sensitivity."""

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    if requested.strip().lower() in lookup:
        return lookup[requested.strip().lower()]

    raise KeyError(
        f"Column '{requested}' not found. "
        f"Available columns: {list(df.columns)}"
    )
def mask_invalid_appliance_readings(df):
    """
    Mask individual appliance readings when REFIT marks
    the household observation as inconsistent.

    Aggregate readings are preserved.
    Invalid appliance readings become NaN, not zero.
    """
    df = df.copy()

    if "Issues" not in df.columns:
        raise KeyError(
            "Issues column not found. "
            "Cannot apply REFIT data-quality filtering."
        )

    issue_mask = pd.to_numeric(
        df["Issues"], errors="coerce"
    ).eq(1)

    appliance_cols = [
        col for col in df.columns
        if str(col).startswith("Appliance")
    ]

    df.loc[issue_mask, appliance_cols] = float("nan")

    return df
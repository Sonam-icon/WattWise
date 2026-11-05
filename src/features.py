
import numpy as np
import pandas as pd


def build_features(
    df: pd.DataFrame,
    aggregate_col: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build time-based and aggregate-load features
    for washing-machine energy disaggregation.
    """

    d = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .copy()
        .set_index("timestamp")
    )

    x = pd.DataFrame(index=d.index)

    # Current aggregate power
    agg = pd.to_numeric(
        d[aggregate_col],
        errors="coerce"
    ).clip(lower=0)

    x["aggregate"] = agg

    # Lag features
    for lag in [1, 2, 3, 5, 10, 15]:
        x[f"agg_lag_{lag}"] = agg.shift(lag)

    # Rolling statistics
    for window in [5, 15, 30]:
        x[f"agg_roll_mean_{window}"] = (
            agg.rolling(window, min_periods=1).mean()
        )

        x[f"agg_roll_max_{window}"] = (
            agg.rolling(window, min_periods=1).max()
        )

        x[f"agg_roll_std_{window}"] = (
            agg.rolling(window, min_periods=2).std()
        )

    # Time-based features
    x["hour"] = x.index.hour
    x["dayofweek"] = x.index.dayofweek
    x["minute_of_day"] = (
        x.index.hour * 60 + x.index.minute
    )

    # Cyclical time features
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)

    x["day_sin"] = np.sin(
        2 * np.pi * x["dayofweek"] / 7
    )
    x["day_cos"] = np.cos(
        2 * np.pi * x["dayofweek"] / 7
    )

    # Replace infinite values
    x = x.replace([np.inf, -np.inf], np.nan)

    return x, x.columns.tolist()

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)

# Candidate power thresholds used to convert the continuous target into
# ON/OFF labels for the two-stage model.
CANDIDATE_THRESHOLDS_W = [5.0, 10.0, 20.0, 30.0]


def metrics(y_true, y_pred):
    """Calculate minute-level regression metrics and total-energy error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    # Each observation represents one minute.
    # W-minutes / 60,000 = kWh.
    energy_true = float(np.sum(y_true) / 60000.0)
    energy_pred = float(np.sum(y_pred) / 60000.0)

    return {
        "MAE_W": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_W": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "actual_energy_kWh": energy_true,
        "predicted_energy_kWh": energy_pred,
        "absolute_energy_error_kWh": float(abs(energy_true - energy_pred)),
    }


def make_classifier():
    """Create the ON/OFF Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=40,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def make_regressor():
    """Create the washing-machine power Random Forest regressor."""
    return RandomForestRegressor(
        n_estimators=50,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )


def select_activity_threshold(
    X_train,
    y_train,
    candidate_thresholds=CANDIDATE_THRESHOLDS_W,
    validation_fraction=0.2,
):
    """
    Select an ON/OFF threshold using an internal chronological split
    of the outer training data only. The outer test set is not used.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    split = int(len(X_train) * (1 - validation_fraction))

    X_inner = X_train.iloc[:split]
    X_val = X_train.iloc[split:]

    y_inner = y_train.iloc[:split]
    y_val = y_train.iloc[split:]

    if len(X_inner) < 100 or len(X_val) < 20:
        raise ValueError("Insufficient data for internal validation.")

    rows = []

    for threshold in candidate_thresholds:
        labels_inner = (y_inner >= threshold).astype(int)
        labels_val = (y_val >= threshold).astype(int)

        if labels_inner.nunique() < 2:
            rows.append({
                "threshold_w": float(threshold),
                "validation_precision": np.nan,
                "validation_recall": np.nan,
                "validation_f1": np.nan,
                "status": "Skipped: inner training has only one class",
            })
            continue

        classifier = make_classifier()
        classifier.fit(X_inner, labels_inner)

        predicted_labels = classifier.predict(X_val)

        rows.append({
            "threshold_w": float(threshold),
            "validation_precision": float(
                precision_score(
                    labels_val,
                    predicted_labels,
                    zero_division=0,
                )
            ),
            "validation_recall": float(
                recall_score(
                    labels_val,
                    predicted_labels,
                    zero_division=0,
                )
            ),
            "validation_f1": float(
                f1_score(
                    labels_val,
                    predicted_labels,
                    zero_division=0,
                )
            ),
            "status": "Evaluated",
        })

    tuning = pd.DataFrame(rows)

    valid = tuning.dropna(subset=["validation_f1"])

    if valid.empty:
        raise ValueError(
            "No candidate threshold produced a valid classifier."
        )

    # Highest validation F1 wins; lower threshold wins an exact tie.
    best = valid.sort_values(
        ["validation_f1", "threshold_w"],
        ascending=[False, True],
    ).iloc[0]

    return float(best["threshold_w"]), tuning


def plot_actual_vs_predicted(predictions, model_name, output_path):
    """
    Save a readable comparison plot.

    The full test predictions remain unchanged. For the detailed plot,
    select a 24-hour test window with the greatest actual washing-machine
    energy so the reviewer can inspect a period containing real activity.
    """
    plot_data = predictions.copy()

    plot_data["timestamp"] = pd.to_datetime(
        plot_data["timestamp"]
    )

    plot_data = (
        plot_data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "Cannot plot an empty prediction dataframe."
        )

    # Use a representative high-activity 24-hour window where possible.
    window_size = min(1440, len(plot_data))

    if len(plot_data) > window_size:
        actual_energy_window = (
            plot_data["actual_washing_machine_w"]
            .rolling(
                window=window_size,
                min_periods=window_size,
            )
            .sum()
        )

        end_position = int(actual_energy_window.idxmax())
        start_position = end_position - window_size + 1

        sample = plot_data.iloc[
            start_position:end_position + 1
        ].copy()

    else:
        sample = plot_data.copy()

    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot predictions first.
    ax.plot(
        sample["timestamp"],
        sample["predicted_washing_machine_w"],
        label=f"Predicted ({model_name})",
        color="darkorange",
        linewidth=1.1,
        alpha=0.8,
        zorder=2,
    )

    # Plot actual values on top so the actual trace is not hidden.
    ax.plot(
        sample["timestamp"],
        sample["actual_washing_machine_w"],
        label="Actual",
        color="royalblue",
        linewidth=1.25,
        alpha=0.95,
        zorder=3,
    )

    ax.set_title(
        "Washing-machine Disaggregation: Actual vs Predicted "
        "(Representative 24-hour Test Window)"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")

    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def train_evaluate(
    df: pd.DataFrame,
    features: pd.DataFrame,
    target_col: str,
    output_dir: str | Path,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """
    Train and compare:

      1. Always-off baseline
      2. Single-stage Random Forest
      3. LightGBM (when installed)
      4. Two-stage Random Forest

    The final test period is chronological and is not used for
    threshold tuning.

    Returns the selected model's test predictions and metrics.
    """

    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' was not found in df."
        )

    if features.empty:
        raise ValueError(
            "The feature dataframe is empty."
        )

    # Prepare target as a timestamp-indexed series.
    target_data = (
        df.copy()
        .assign(
            timestamp=pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )
        )
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )

    y = pd.to_numeric(
        target_data[target_col],
        errors="coerce",
    ).clip(lower=0)

    y.name = "target"

    # Features are expected to have a DatetimeIndex.
    X = features.copy()

    X.index = pd.to_datetime(
        X.index,
        errors="coerce",
    )

    X = X.loc[~X.index.isna()]
    X = X.sort_index()
    X = X.loc[~X.index.duplicated(keep="last")]

    data = X.join(y, how="inner")

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    if data.empty:
        raise ValueError(
            "No valid observations remain after aligning features "
            "and target. Check that features use a timestamp index "
            "matching df['timestamp']."
        )

    # Chronological outer holdout.
    split = int(len(data) * (1 - test_fraction))

    train = data.iloc[:split]
    test = data.iloc[split:]

    if len(train) < 200 or len(test) < 20:
        raise ValueError(
            "Insufficient data for the chronological train/test split."
        )

    feature_columns = list(features.columns)

    X_train = train[feature_columns]
    y_train = train["target"]

    X_test = test[feature_columns]
    y_test = test["target"]

    result = {
        "split": {
            "strategy": "chronological holdout",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start": str(train.index.min()),
            "train_end": str(train.index.max()),
            "test_start": str(test.index.min()),
            "test_end": str(test.index.max()),
        }
    }

    model_predictions = {}
    metric_keys = {}

    # ---------------------------------------------------------
    # 1. Always-off baseline
    # ---------------------------------------------------------

    baseline = DummyRegressor(
        strategy="constant",
        constant=0.0,
    )

    baseline.fit(X_train, y_train)

    result["baseline_always_off"] = metrics(
        y_test,
        baseline.predict(X_test),
    )

    # ---------------------------------------------------------
    # 2. Single-stage Random Forest
    # ---------------------------------------------------------

    random_forest = make_regressor()

    random_forest.fit(
        X_train,
        y_train,
    )

    pred_rf = np.clip(
        random_forest.predict(X_test),
        0,
        None,
    )

    result["random_forest"] = metrics(
        y_test,
        pred_rf,
    )

    model_predictions["Random Forest"] = pred_rf
    metric_keys["Random Forest"] = "random_forest"

    # ---------------------------------------------------------
    # 3. LightGBM
    # ---------------------------------------------------------

    try:
        from lightgbm import LGBMRegressor

        lightgbm = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

        lightgbm.fit(
            X_train,
            y_train,
        )

        pred_lgbm = np.clip(
            lightgbm.predict(X_test),
            0,
            None,
        )

        result["lightgbm"] = metrics(
            y_test,
            pred_lgbm,
        )

        model_predictions["LightGBM"] = pred_lgbm
        metric_keys["LightGBM"] = "lightgbm"

    except ImportError:
        result["lightgbm"] = (
            "Skipped: LightGBM is not installed."
        )

    # ---------------------------------------------------------
    # 4. Two-stage Random Forest
    # ---------------------------------------------------------

    selected_threshold, tuning_df = select_activity_threshold(
        X_train,
        y_train,
        candidate_thresholds=CANDIDATE_THRESHOLDS_W,
        validation_fraction=0.2,
    )

    result["threshold_selection"] = {
        "method": (
            "Internal chronological validation; highest validation F1"
        ),
        "candidate_thresholds_w": CANDIDATE_THRESHOLDS_W,
        "selected_threshold_w": selected_threshold,
        "validation_fraction_of_outer_training_data": 0.2,
    }

    y_train_active = (
        y_train >= selected_threshold
    ).astype(int)

    y_test_active = (
        y_test >= selected_threshold
    ).astype(int)

    if y_train_active.nunique() != 2:
        raise ValueError(
            "Selected threshold produced only one activity class "
            "in training."
        )

    classifier = make_classifier()

    classifier.fit(
        X_train,
        y_train_active,
    )

    pred_active = classifier.predict(X_test)

    # Train power regressor only on active training observations.
    active_mask = y_train >= selected_threshold

    if int(active_mask.sum()) < 100:
        raise ValueError(
            "Insufficient active training observations "
            "for power regression."
        )

    active_regressor = make_regressor()

    active_regressor.fit(
        X_train.loc[active_mask],
        y_train.loc[active_mask],
    )

    estimated_power = np.clip(
        active_regressor.predict(X_test),
        0,
        None,
    )

    # If classifier predicts OFF, estimated appliance power is zero.
    pred_two_stage = np.where(
        pred_active == 1,
        estimated_power,
        0.0,
    )

    result["two_stage_random_forest"] = metrics(
        y_test,
        pred_two_stage,
    )

    result["two_stage_activity_detection"] = {
        "precision": float(
            precision_score(
                y_test_active,
                pred_active,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test_active,
                pred_active,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test_active,
                pred_active,
                zero_division=0,
            )
        ),
        "actual_active_minutes": int(
            y_test_active.sum()
        ),
        "predicted_active_minutes": int(
            pred_active.sum()
        ),
    }

    model_predictions["Two-stage Random Forest"] = pred_two_stage
    metric_keys["Two-stage Random Forest"] = (
        "two_stage_random_forest"
    )

    # ---------------------------------------------------------
    # Leak-Free Model Selection on Internal Validation Split
    # ---------------------------------------------------------
    val_split = int(len(X_train) * 0.8)
    X_inner = X_train.iloc[:val_split]
    y_inner = y_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]

    val_metrics = {}

    # Validation: Random Forest
    val_rf = make_regressor()
    val_rf.fit(X_inner, y_inner)
    val_metrics["Random Forest"] = metrics(
        y_val, np.clip(val_rf.predict(X_val), 0, None)
    )["absolute_energy_error_kWh"]

    # Validation: LightGBM (if available)
    if "LightGBM" in model_predictions:
        try:
            val_lgbm = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )
            val_lgbm.fit(X_inner, y_inner)
            val_metrics["LightGBM"] = metrics(
                y_val, np.clip(val_lgbm.predict(X_val), 0, None)
            )["absolute_energy_error_kWh"]
        except Exception:
            pass

    # Validation: Two-stage Random Forest
    val_clf = make_classifier()
    val_clf.fit(X_inner, (y_inner >= selected_threshold).astype(int))
    pred_val_active = val_clf.predict(X_val)

    val_act_mask = y_inner >= selected_threshold
    val_act_reg = make_regressor()
    val_act_reg.fit(X_inner.loc[val_act_mask], y_inner.loc[val_act_mask])
    pred_val_two_stage = np.where(
        pred_val_active == 1,
        np.clip(val_act_reg.predict(X_val), 0, None),
        0.0,
    )
    val_metrics["Two-stage Random Forest"] = metrics(
        y_val, pred_val_two_stage
    )["absolute_energy_error_kWh"]

    # Select best model strictly using validation split
    best_name = min(val_metrics, key=val_metrics.get)
    best_prediction = model_predictions[best_name]

    result["selected_model"] = best_name
    result["model_selection_criterion"] = (
        "Lowest absolute total energy estimation error on chronological validation split (leak-free)"
    )
    result["validation_energy_error_comparison_kWh"] = val_metrics

    # ---------------------------------------------------------
    # Save predictions and evaluation outputs
    # ---------------------------------------------------------

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.DataFrame({
        "timestamp": test.index,
        "actual_washing_machine_w": y_test.to_numpy(),
        "predicted_washing_machine_w": best_prediction,
    })

    predictions.to_csv(
        output_path / "test_predictions.csv",
        index=False,
    )

    tuning_df.to_csv(
        output_path / "threshold_tuning.csv",
        index=False,
    )

    # Generate the improved Actual vs Predicted graph.
    plot_actual_vs_predicted(
        predictions,
        best_name,
        output_path / "actual_vs_predicted.png",
    )

    # Save model comparison, including the always-off baseline.
    comparison_rows = []

    for model_name in model_predictions:
        comparison_rows.append({
            "model": model_name,
            **result[metric_keys[model_name]],
        })

    comparison_rows.append({
        "model": "Always-off baseline",
        **result["baseline_always_off"],
    })

    pd.DataFrame(comparison_rows).to_csv(
        output_path / "model_comparison.csv",
        index=False,
    )

    return predictions, result
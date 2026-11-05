
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.dummy import DummyRegressor


CANDIDATE_THRESHOLDS_W = [5.0, 10.0, 20.0, 30.0]


def metrics(y_true, y_pred):
    """Calculate regression and energy estimation metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    energy_true = float(np.sum(y_true) / 60000.0)
    energy_pred = float(np.sum(y_pred) / 60000.0)

    return {
        "MAE_W": float(mae),
        "RMSE_W": float(rmse),
        "actual_energy_kWh": energy_true,
        "predicted_energy_kWh": energy_pred,
        "absolute_energy_error_kWh": float(
            abs(energy_true - energy_pred)
        ),
    }


def make_classifier(n_estimators=50):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=16,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def make_regressor(n_estimators=50):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=16,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
    )


def select_activity_threshold(
    X_train,
    y_train,
    candidate_thresholds=CANDIDATE_THRESHOLDS_W,
    validation_fraction=0.2,
):
    """Select activity threshold using chronological validation."""

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
                "status": "Skipped: only one training class",
            })
            continue

        classifier = make_classifier(n_estimators=40)

        classifier.fit(X_inner, labels_inner)
        predictions = classifier.predict(X_val)

        rows.append({
            "threshold_w": float(threshold),
            "validation_precision": float(
                precision_score(
                    labels_val, predictions, zero_division=0
                )
            ),
            "validation_recall": float(
                recall_score(
                    labels_val, predictions, zero_division=0
                )
            ),
            "validation_f1": float(
                f1_score(
                    labels_val, predictions, zero_division=0
                )
            ),
            "status": "Evaluated",
        })

    tuning = pd.DataFrame(rows)
    valid = tuning.dropna(subset=["validation_f1"])

    if valid.empty:
        raise ValueError("No candidate threshold produced a valid classifier.")

    best = valid.sort_values(
        ["validation_f1", "threshold_w"],
        ascending=[False, True],
    ).iloc[0]

    return float(best["threshold_w"]), tuning


def train_evaluate(
    df: pd.DataFrame,
    features: pd.DataFrame,
    target_col: str,
    output_dir: str | Path,
    test_fraction: float = 0.2,
):
    """Train and compare baseline, Random Forest, LightGBM and two-stage RF."""

    print("Preparing model data...", flush=True)

    target_data = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )

    y = pd.to_numeric(
        target_data[target_col],
        errors="coerce",
    ).clip(lower=0)

    data = features.join(y.rename("target"), how="inner")

    # Missing targets and missing features are excluded, never converted to zero.
    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    if data.empty:
        raise ValueError("No valid observations remain after alignment.")

    split = int(len(data) * (1 - test_fraction))

    train = data.iloc[:split]
    test = data.iloc[split:]

    if len(train) < 200 or len(test) < 20:
        raise ValueError("Insufficient data for chronological split.")

    X_train = train[features.columns]
    y_train = train["target"]

    X_test = test[features.columns]
    y_test = test["target"]

    print(
        f"Training rows: {len(train):,} | Test rows: {len(test):,}",
        flush=True,
    )

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

    # Always-off baseline
    print("Evaluating always-off baseline...", flush=True)

    baseline = DummyRegressor(strategy="constant", constant=0.0)
    baseline.fit(X_train, y_train)

    pred_base = baseline.predict(X_test)

    result["baseline_always_off"] = metrics(y_test, pred_base)

    # Single-stage Random Forest
    print("Training Random Forest regression...", flush=True)

    rf = make_regressor(n_estimators=50)
    rf.fit(X_train, y_train)

    pred_rf = np.clip(rf.predict(X_test), 0, None)

    result["random_forest"] = metrics(y_test, pred_rf)
    model_predictions["Random Forest"] = pred_rf

    # LightGBM
    try:
        from lightgbm import LGBMRegressor

        print("Training LightGBM...", flush=True)

        lgbm = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        lgbm.fit(X_train, y_train)
        pred_lgbm = np.clip(lgbm.predict(X_test), 0, None)

        result["lightgbm"] = metrics(y_test, pred_lgbm)
        model_predictions["LightGBM"] = pred_lgbm

    except ImportError:
        result["lightgbm"] = "Skipped: LightGBM is not installed."

    # Threshold selection
    print("Selecting activity threshold...", flush=True)

    selected_threshold, tuning_df = select_activity_threshold(
        X_train,
        y_train,
        candidate_thresholds=CANDIDATE_THRESHOLDS_W,
        validation_fraction=0.2,
    )

    result["threshold_selection"] = {
        "method": "Internal chronological validation; highest validation F1",
        "candidate_thresholds_w": CANDIDATE_THRESHOLDS_W,
        "selected_threshold_w": selected_threshold,
        "validation_fraction_of_outer_training_data": 0.2,
    }

    # Two-stage Random Forest
    print("Training two-stage activity classifier...", flush=True)

    y_train_active = (y_train >= selected_threshold).astype(int)
    y_test_active = (y_test >= selected_threshold).astype(int)

    if y_train_active.nunique() != 2:
        raise ValueError(
            "Selected threshold produced only one training class."
        )

    classifier = make_classifier(n_estimators=50)
    classifier.fit(X_train, y_train_active)

    pred_active = classifier.predict(X_test)

    print("Training active-power regressor...", flush=True)

    active_mask = y_train >= selected_threshold

    if int(active_mask.sum()) < 100:
        raise ValueError("Insufficient active observations for power regression.")

    regressor = make_regressor(n_estimators=50)

    regressor.fit(
        X_train.loc[active_mask],
        y_train.loc[active_mask],
    )

    estimated_power = np.clip(regressor.predict(X_test), 0, None)

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
            precision_score(y_test_active, pred_active, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test_active, pred_active, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test_active, pred_active, zero_division=0)
        ),
        "actual_active_minutes": int(y_test_active.sum()),
        "predicted_active_minutes": int(pred_active.sum()),
    }

    model_predictions["Two-stage Random Forest"] = pred_two_stage

    # Select best model by energy estimation error
    metric_keys = {
        "Random Forest": "random_forest",
        "LightGBM": "lightgbm",
        "Two-stage Random Forest": "two_stage_random_forest",
    }

    best_name = min(
        model_predictions,
        key=lambda name: result[
            metric_keys[name]
        ]["absolute_energy_error_kWh"],
    )

    best_pred = model_predictions[best_name]

    result["selected_model"] = best_name
    result["model_selection_criterion"] = (
        "Lowest absolute total energy estimation error"
    )

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "timestamp": test.index,
        "actual_washing_machine_w": y_test.values,
        "predicted_washing_machine_w": best_pred,
    })

    predictions.to_csv(out / "test_predictions.csv", index=False)
    tuning_df.to_csv(out / "threshold_tuning.csv", index=False)

    # Actual vs predicted plot
    print("Saving evaluation plots and metrics...", flush=True)

    fig, ax = plt.subplots(figsize=(12, 4))

    sample = predictions.head(min(1500, len(predictions)))

    ax.plot(
        sample["timestamp"],
        sample["actual_washing_machine_w"],
        label="Actual",
        linewidth=1,
    )

    ax.plot(
        sample["timestamp"],
        sample["predicted_washing_machine_w"],
        label=f"Predicted ({best_name})",
        linewidth=1,
    )

    ax.set_title(
        "Washing-machine disaggregation: Actual vs Predicted"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # Model comparison
    comparison = []

    for name in model_predictions:
        comparison.append({
            "model": name,
            **result[metric_keys[name]],
        })

    comparison.append({
        "model": "Always-off baseline",
        **result["baseline_always_off"],
    })

    pd.DataFrame(comparison).to_csv(
        out / "model_comparison.csv",
        index=False,
    )

    print("Model training and evaluation completed.", flush=True)

    return predictions, result

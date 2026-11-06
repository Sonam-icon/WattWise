"""
Unit and regression tests for WattWise pipeline modules.
Uses standard library unittest so it runs in any standard Python environment.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Mock matplotlib if not installed so tests run in minimal CI/Python environments
try:
    import matplotlib
except ImportError:
    sys.modules["matplotlib"] = MagicMock()
    sys.modules["matplotlib.pyplot"] = MagicMock()

import numpy as np
import pandas as pd

from src.io_utils import (
    resolve_column,
    mask_invalid_appliance_readings,
)
from src.clean import (
    count_missing_runs,
    interpolate_short_internal_gaps,
)
from src.eda import detect_cycles, household_summary
from src.features import build_features
from src.model import metrics, make_classifier, make_regressor


class TestIOUtils(unittest.TestCase):
    def test_resolve_column(self):
        df = pd.DataFrame(columns=["Aggregate", "Appliance1", "Issues"])
        self.assertEqual(resolve_column(df, "aggregate"), "Aggregate")
        self.assertEqual(resolve_column(df, "APPLIANCE1"), "Appliance1")
        with self.assertRaises(KeyError):
            resolve_column(df, "NonExistent")

    def test_mask_invalid_appliance_readings(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="min"),
            "Aggregate": [100.0, 200.0, 300.0, 400.0],
            "Appliance1": [50.0, 60.0, 70.0, 80.0],
            "Appliance2": [10.0, 20.0, 30.0, 40.0],
            "Issues": [0, 1, 0, 1],
        })
        masked = mask_invalid_appliance_readings(df)

        # Aggregate must NOT be masked
        np.testing.assert_array_equal(masked["Aggregate"].values, [100.0, 200.0, 300.0, 400.0])

        # Rows 1 and 3 (where Issues == 1) must have NaN for appliances
        self.assertEqual(masked.loc[0, "Appliance1"], 50.0)
        self.assertTrue(np.isnan(masked.loc[1, "Appliance1"]))
        self.assertEqual(masked.loc[2, "Appliance1"], 70.0)
        self.assertTrue(np.isnan(masked.loc[3, "Appliance1"]))


class TestClean(unittest.TestCase):
    def test_count_missing_runs(self):
        s1 = pd.Series([1, 2, 3, 4])
        self.assertEqual(count_missing_runs(s1), 0)

        s2 = pd.Series([np.nan, np.nan, 1, 2, np.nan, 3, np.nan, np.nan])
        # Three missing runs: at start (len 2), in middle (len 1), at end (len 2)
        self.assertEqual(count_missing_runs(s2), 3)

    def test_interpolate_short_internal_gaps(self):
        # 1-minute time series with:
        # - Leading boundary gap of 1 minute (should NOT be interpolated)
        # - Internal gap of 2 minutes (should BE interpolated)
        # - Internal gap of 4 minutes (exceeds max_gap_minutes=3, should NOT be interpolated)
        # - Trailing boundary gap of 1 minute (should NOT be interpolated)
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="1min")
        values = [np.nan, 10.0, np.nan, np.nan, 40.0, np.nan, np.nan, np.nan, np.nan, 90.0, 100.0, np.nan]
        df = pd.DataFrame({"power": values}, index=idx)

        res = interpolate_short_internal_gaps(df, max_gap_minutes=3)

        # Leading boundary remains NaN
        self.assertTrue(np.isnan(res.iloc[0, 0]))

        # Internal 2-minute gap (indices 2 and 3) should be interpolated linearly between 10.0 and 40.0
        self.assertAlmostEqual(res.iloc[2, 0], 20.0)
        self.assertAlmostEqual(res.iloc[3, 0], 30.0)

        # Internal 4-minute gap (indices 5, 6, 7, 8) exceeds 3 minutes and must remain NaN
        self.assertTrue(np.isnan(res.iloc[5, 0]))
        self.assertTrue(np.isnan(res.iloc[6, 0]))
        self.assertTrue(np.isnan(res.iloc[7, 0]))
        self.assertTrue(np.isnan(res.iloc[8, 0]))

        # Trailing boundary remains NaN
        self.assertTrue(np.isnan(res.iloc[11, 0]))


class TestEDA(unittest.TestCase):
    def test_detect_cycles_basic(self):
        # 10 minutes of operation:
        # Minutes 0..1: idle (0 W)
        # Minutes 2..7: active (100 W) -> 6 minutes duration >= 5 min threshold
        # Minutes 8..9: idle (0 W)
        ts = pd.date_range("2024-01-01 10:00", periods=10, freq="1min")
        power = [0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0]
        df = pd.DataFrame({"timestamp": ts, "wm": power})

        cycles = detect_cycles(df, "wm", threshold_w=5.0, min_duration_minutes=5)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles.iloc[0]["duration_min"], 6.0)
        self.assertEqual(cycles.iloc[0]["peak_power_w"], 100.0)
        # Energy = 6 minutes * 100 W / 60,000 = 0.01 kWh
        self.assertAlmostEqual(cycles.iloc[0]["energy_kwh"], 0.01)

    def test_detect_cycles_rejects_short_pulses(self):
        # 3-minute pulse < 5-minute min_duration_minutes
        ts = pd.date_range("2024-01-01 10:00", periods=6, freq="1min")
        power = [0.0, 2000.0, 2000.0, 2000.0, 0.0, 0.0]
        df = pd.DataFrame({"timestamp": ts, "wm": power})

        cycles = detect_cycles(df, "wm", threshold_w=5.0, min_duration_minutes=5)
        self.assertTrue(cycles.empty)

    def test_detect_cycles_merges_short_gaps(self):
        # Active 3 min, 2 min pause (0 W), active 3 min -> total duration = 8 min >= 5 min
        ts = pd.date_range("2024-01-01 10:00", periods=8, freq="1min")
        power = [500.0, 500.0, 500.0, 0.0, 0.0, 500.0, 500.0, 500.0]
        df = pd.DataFrame({"timestamp": ts, "wm": power})

        cycles = detect_cycles(df, "wm", threshold_w=5.0, min_duration_minutes=5, merge_gap_minutes=3)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles.iloc[0]["duration_min"], 8.0)

    def test_detect_cycles_does_not_bridge_nan(self):
        # Active 3 min, then NaN (break in data), then active 3 min
        # Since NaN breaks continuity, neither segment reaches 5 minutes
        ts = pd.date_range("2024-01-01 10:00", periods=7, freq="1min")
        power = [500.0, 500.0, 500.0, np.nan, 500.0, 500.0, 500.0]
        df = pd.DataFrame({"timestamp": ts, "wm": power})

        cycles = detect_cycles(df, "wm", threshold_w=5.0, min_duration_minutes=5)
        self.assertTrue(cycles.empty)


class TestFeatures(unittest.TestCase):
    def test_build_features(self):
        ts = pd.date_range("2024-01-01 00:00", periods=60, freq="1min")
        agg = np.random.uniform(100, 3000, size=60)
        df = pd.DataFrame({"timestamp": ts, "Aggregate": agg})

        X, feature_names = build_features(df, "Aggregate")

        self.assertEqual(len(feature_names), 23)  # 1 agg + 6 lags + 9 rolling + 7 time/cyclical
        self.assertIn("aggregate", feature_names)
        self.assertIn("agg_lag_1", feature_names)
        self.assertIn("agg_roll_mean_5", feature_names)
        self.assertIn("hour_sin", feature_names)
        self.assertEqual(len(X), 60)


class TestModel(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = [0.0, 0.0, 600.0, 600.0]  # Total energy = 1,200 W-min / 60,000 = 0.02 kWh
        y_pred = [0.0, 0.0, 0.0, 0.0]      # Baseline always-off
        res = metrics(y_true, y_pred)

        self.assertAlmostEqual(res["MAE_W"], 300.0)
        self.assertAlmostEqual(res["actual_energy_kWh"], 0.02)
        self.assertAlmostEqual(res["predicted_energy_kWh"], 0.00)
        self.assertAlmostEqual(res["absolute_energy_error_kWh"], 0.02)


if __name__ == "__main__":
    unittest.main()

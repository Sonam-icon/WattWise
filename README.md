# Flock Energy | REFIT Data Cleaning, Washing-Machine EDA & Disaggregation

This project addresses the Flock Energy data-science take-home assignment using the REFIT Electrical Load Measurements dataset. It covers raw-data inspection and cleaning, one-minute washing-machine usage analysis across four households, and a supervised appliance-disaggregation experiment using household aggregate power.

## Project scope

1. **Raw-data cleaning (House 1):** timestamp and sampling checks, duplicate handling, negative-reading treatment, one-minute resampling, limited short-gap interpolation, quality reporting and before/after visualisation.
2. **Washing-machine EDA (Houses 2, 3, 5 and 6):** one-minute usage profiles, missingness and off-time summaries, operating-event detection, hourly/weekly patterns and cross-house cycle comparisons.
3. **Disaggregation (House 5):** estimate washing-machine power from aggregate household power using a chronological holdout, an always-off baseline, Random Forest, LightGBM and a two-stage Random Forest.

## Repository structure

```text
Flock_Energy_Assignment/
├── config/
│   ├── household_channels.example.json
│   └── household_channels.json       # local only; do not commit if it contains local paths
├── data/
│   ├── Raw/                          # local REFIT raw files; not committed
│   ├── Cleaned/                      # local cleaned REFIT files; not committed
│   └── metadata/                     # dataset readme/metadata
├── notebooks/
├── outputs/                          # generated reports and plots; selected plots may be copied to docs/assets
├── src/
│   ├── io_utils.py
│   ├── clean.py
│   ├── eda.py
│   ├── features.py
│   └── model.py
├── run_pipeline.py
├── requirements.txt
├── DATA.md
├── RESULTS.md
└── Recommendation.md
```

## Dataset

The project uses the University of Strathclyde's REFIT Electrical Load Measurements dataset. The raw and cleaned data are not included in this repository because the source files are large and are distributed under the dataset provider's terms.

- [Official REFIT dataset page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)
- [Raw REFIT archive](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z)
- [Cleaned REFIT archive](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z)
- [Cleaned REFIT readme](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)

Download the dataset from the official source and place the relevant files in the local `data/Raw/` and `data/Cleaned/` directories. See [DATA.md](DATA.md) for the household coverage, channel mapping and data-quality decisions. Do not assume that the same `ApplianceN` column represents the same appliance in every home; use the documented household-specific mapping.

## Setup (Windows CMD)

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the local channel configuration using the example file, then verify it against the cleaned REFIT metadata:

```cmd
copy config\household_channels.example.json config\household_channels.json
```

Ensure the local data files and configuration match the paths and column names expected by `run_pipeline.py`. Run the complete workflow:

```cmd
python run_pipeline.py
```

Generated tables, model metrics and plots are written under `outputs/`. Exact output names are determined by the pipeline scripts.

## Method overview

### Raw cleaning

- Parse timestamps using the source timestamp format and sort chronologically.
- Measure duplicate timestamps and sampling-interval irregularity before cleaning.
- Keep the last record for duplicate timestamps.
- Treat negative power readings as missing rather than as zero.
- Resample power channels to one-minute means.
- Interpolate only short internal gaps of up to three consecutive minutes; longer and boundary gaps remain missing.
- Flag high readings for review rather than automatically deleting them without evidence.

The cleaned REFIT release is itself preprocessed by its maintainers. The raw-data workflow in this project should therefore be understood as additional inspection and processing of the supplied raw release, not as a reconstruction of the original REFIT cleaning process.

### Washing-machine event rule

An appliance reading of at least **5 W** is treated as active. An event must contain at least **5 consecutive active minutes**. Off-gaps of up to **3 minutes** are merged when they occur within a known-data segment. Missing readings break events; they are not silently treated as zero or as continuous operation. These are operational event definitions for comparison, not verified physical wash-program boundaries.

### Disaggregation model

The target is House 5's washing-machine channel (`Appliance3`), and the input is the household aggregate power signal. Features use current and historical aggregate readings (lags and rolling statistics) plus time-of-day/day-of-week encodings. The evaluation uses a chronological 80:20 holdout; the final test period is later than the training period. Threshold selection is performed on an internal chronological validation portion of the training data.

The selected model is the two-stage Random Forest because it has the lowest absolute total-energy estimation error among the evaluated models. It is not the best model on every metric; see [RESULTS.md](RESULTS.md).

## Reproducibility and interpretation

- Use the same official REFIT release and household-specific channel mappings to reproduce the reported results.
- The results are household- and period-specific; the House 5 model is not evidence of generalisation to every home.
- Missingness, appliance overlap, behavioural differences and the operational 5 W threshold affect the results.
- Estimated appliance energy is not a verified appliance sub-meter reading or a billing-grade measurement.

## GitHub hygiene

Do **not** commit raw/cleaned REFIT CSV files, downloaded archives, `.venv/`, caches, or large intermediate files. Commit source code, documentation, the example configuration and small metadata needed to reproduce the workflow. Only add selected plots/results if they are permitted by the dataset terms and do not make the repository unnecessarily large. Keep any real local configuration file out of Git if it contains machine-specific paths.

## AI assistance

If AI tools were used during development, describe their role accurately in the submission. Review and run the code yourself; do not claim independent validation beyond the checks actually performed.

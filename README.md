# WattWise: Understanding Household Energy Use and Washing-Machine Consumption

## About the project

WattWise is my submission for the Flock Energy data science assignment.
I used the REFIT Electrical Load Measurements dataset to explore how
households use electricity, understand the behaviour of washing
machines, and investigate how accurately washing-machine consumption can
be estimated from a home's overall electricity demand.

I approached the work in three parts. First, I examined the quality of
the supplied measurements and carried out additional cleaning and
resampling. Next, I analysed washing-machine usage across four
households. Finally, I built and evaluated machine-learning models to
estimate the washing-machine load from aggregate household power.

The analysis focuses on the data available in REFIT. The detected
operating events should therefore be interpreted as data-driven
estimates, rather than confirmed wash-program boundaries.

## What I investigated

### 1. Data quality and preprocessing

I began by examining the House 1 measurements to understand the
structure and reliability of the time-series data. The inspection
covered timestamp validity, duplicate timestamps, irregular sampling
intervals, missing observations and unusual power readings.

For the additional preprocessing, I sorted the observations by
timestamp, handled duplicate timestamps, treated negative power readings
as missing, and resampled numeric power readings to one-minute averages.
I limited interpolation to short internal gaps of up to three
consecutive minutes. Longer gaps and gaps at the beginning or end of a
series were left missing so that the processing would not create
unsupported consumption values.

The REFIT cleaned release has already undergone preprocessing by its
maintainers. My House 1 work is therefore an additional inspection and
processing step, not a recreation of the original REFIT cleaning
procedure.

For the household analysis, I also considered the REFIT `Issues`
indicator. Where a row was flagged, I masked the appliance readings
while retaining the aggregate reading. I used this as a conservative
quality-control decision; it does not prove that every appliance
measurement in every flagged row is individually incorrect.

### 2. Washing-machine usage across households

I analysed Houses 2, 3, 5 and 6 at one-minute resolution. For each
household, I examined the available measurement period, missing data,
the proportion of time the washing-machine channel appeared inactive,
and the operating events detected from the appliance power readings.

To make the event analysis consistent across homes, I treated readings
of at least **5 W** as active. An event needed at least **5 active
minutes**, and inactive gaps of up to **3 minutes** were merged when
they occurred within a valid-data segment. Missing readings were treated
as breaks in the data rather than as zero-power observations.

These thresholds provide a practical and consistent way to identify
operating events in the available measurements. They should not be
interpreted as a guarantee that each detected event corresponds exactly
to one complete washing programme.

I then looked at how usage varied by hour and day of the week, and
compared the detected cycle characteristics across the four households.
The results and supporting figures are presented in
[RESULTS.md](RESULTS.md).

### 3. Estimating washing-machine load from aggregate power

For the disaggregation experiment, I selected House 5 and used its
`Appliance3` channel as the washing-machine target. The model input was
the household aggregate power reading. This allowed me to test whether
the washing-machine signal could be estimated from the overall household
load.

I created features from the current aggregate reading, lagged readings,
rolling statistics and time information. I compared an always-off
baseline with Random Forest, LightGBM and a two-stage Random Forest
approach.

To avoid using future observations to predict the past, I used a
chronological 80:20 train/test split. Threshold selection for the
two-stage approach was carried out using an internal chronological
validation portion of the training data. The test period was kept
separate for final evaluation.

## What the results show

The two-stage Random Forest produced the lowest absolute total-energy
estimation error among the evaluated machine-learning models. On the
House 5 test period, the measured washing-machine energy was
approximately **85.59 kWh**, while the model estimated approximately
**89.35 kWh**. This corresponds to an absolute error of approximately
**3.76 kWh**, or **4.39%**.

That result needs to be considered alongside the other metrics. Random
Forest achieved lower MAE and RMSE than the other machine-learning
models. The always-off baseline also had a lower MAE than the
machine-learning models because many one-minute washing-machine readings
were zero. However, the always-off baseline estimated no washing-machine
energy at all.

I selected the two-stage Random Forest on the basis of total-energy
estimation error, not because it was best on every metric. Its activity
detection was still imperfect: test precision was approximately
**0.508**, recall was approximately **0.495**, and F1-score was
approximately **0.501**. The model also under-estimated or missed some
high-power events. These results show the difficulty of recovering an
individual appliance's behaviour from aggregate household consumption,
particularly when appliance activity is sparse and different loads can
overlap.

The detailed metrics, figures and examples of model errors are available
in [RESULTS.md](RESULTS.md). The recommendations arising from the
analysis are in [Recommendation.md](Recommendation.md).

## Main limitations and what I would improve next

This experiment is limited to the selected households, appliance
channels and measurement periods. In particular, the disaggregation
model was trained and tested on different time periods from the same
household; it was not evaluated on a completely unseen household. The
reported performance should not be treated as evidence that the model
will transfer directly to other homes.

If I extended the project, I would prioritise:

1.  **Testing generalisation across homes:** Train on multiple
    households and evaluate on a household that was not used during
    training.
2.  **Evaluating complete operating events:** Add event-level matching,
    precision, recall and cycle-level energy error to complement
    minute-level metrics.
3.  **Improving activity detection:** Explore alternative thresholds,
    classifiers and event post-processing to improve the balance between
    false detections and missed activity.
4.  **Trying time-series models:** Compare the tree-based models with
    sequence-based approaches and established non-intrusive load
    monitoring methods.
5.  **Improving temporal features:** Test longer historical windows and
    additional usage-pattern features.
6.  **Studying missing-data effects:** Measure how longer gaps and
    incomplete appliance readings affect cycle detection and
    disaggregation accuracy.

## Dataset and project files

I used the University of Strathclyde's [REFIT Electrical Load
Measurements
dataset](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/).
The raw and cleaned measurement files are not included in this public
repository. They can be obtained from the official sources below,
subject to the dataset's terms of use.

-   [Official REFIT dataset
    page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)
-   [Raw REFIT
    archive](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z)
-   [Cleaned REFIT
    archive](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z)
-   [Cleaned REFIT
    readme](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)

The repository is organised as follows:

``` text
WattWise/
├── config/
│   └── household_channels.json
├── data/
│   └── metadata/
├── docs/
│   └── assets/
├── notebooks/
│   └── analysis.ipynb
├── scripts/
│   └── investigate_extreme_readings.py
├── src/
│   ├── __init__.py
│   ├── clean.py
│   ├── eda.py
│   ├── features.py
│   ├── io_utils.py
│   └── model.py
├── .gitignore
├── DATA.md
├── README.md
├── RESULTS.md
├── Recommendation.md
├── requirements.txt
├── download_data.py
├── inspect_data.py
├── check_timestamps.py
└── run_pipeline.py
```

The `src/` directory contains the main cleaning, EDA, feature
engineering, data I/O and modelling modules. The notebook contains the
analysis workflow, while the scripts provide supporting data inspection
and execution utilities. Selected figures used in the documentation are
in `docs/assets/`.

## Reproducing the analysis

The following setup was written for Windows CMD. The REFIT data files
need to be downloaded separately because they are not stored in this
repository.

``` cmd
git clone https://github.com/Sonam-icon/WattWise.git
cd WattWise

py -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

After installing the dependencies, I place the required REFIT files in
the local `data/Raw/` and `data/Cleaned/` directories, as applicable to
the workflow. I then check `config/household_channels.json` to make sure
the aggregate and washing-machine channel names match the files being
used.

The main workflow can be run with:

``` cmd
python run_pipeline.py
```

Generated reports, tables, model metrics and plots are written to
`outputs/`. The raw data, cleaned data, virtual environment, caches and
generated outputs are excluded from Git. This keeps the public
repository focused on the code, documentation, configuration, metadata
and selected figures rather than redistributing the REFIT measurement
files.

## AI assistance

I used ChatGPT as a development assistant to help interpret the
assignment requirements, discuss implementation and modelling
approaches, troubleshoot errors and improve the project documentation. I
reviewed and adapted suggestions during development. The code was run
locally, and the reported metrics and figures were taken from the
project's generated outputs.

I take responsibility for the final implementation, technical decisions,
interpretation and submitted results.

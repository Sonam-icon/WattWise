# DATA — source, coverage, quality and processing

## Dataset source

This project uses the University of Strathclyde **REFIT Electrical Load Measurements** dataset. It contains household aggregate electricity demand and appliance-level submeter readings.

- [Official REFIT dataset page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)
- [Raw REFIT archive](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z)
- [Cleaned REFIT archive](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z)
- [Cleaned-data metadata/readme](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)

The REFIT cleaned release is already preprocessed by the dataset maintainers (including documented corrections and cleaning). The additional cleaning in this project is not a claim to reproduce all upstream REFIT processing. Record the actual download date and archive version in the local experiment log if the work is rerun.

## Household and period coverage

| Analysis | Household(s) | Sampling | Observed period |
|---|---|---|---|
| Raw-data inspection and cleaning | House 1 | Raw source readings, then 1-minute | 2013-10-09 14:06:17 to 2015-07-10 11:56:32 |
| Washing-machine EDA | Houses 2, 3, 5 and 6 | 1-minute | House 2: 2013-09-17 22:08 to 2015-05-28 08:05; House 3: 2013-09-25 19:21 to 2015-06-02 10:55; House 5: 2013-09-26 09:56 to 2015-07-06 17:48; House 6: 2013-11-28 12:15 to 2015-06-28 22:14 |
| Washing-machine disaggregation | House 5 | 1-minute | 2013-09-26 09:56 to 2015-07-06 17:48 |

The four EDA households have the following washing-machine channel mapping:

| Household | Aggregate channel | Washing-machine channel |
|---|---|---|
| 2 | `Aggregate` | `Appliance2` |
| 3 | `Aggregate` | `Appliance6` |
| 5 | `Aggregate` | `Appliance3` |
| 6 | `Aggregate` | `Appliance2` |

The mapping is household-specific. House 5's `Appliance3` is identified as the washing-machine channel in the project configuration and REFIT appliance metadata.

## House 1 raw-data inspection

The inspected House 1 raw data contained **8,518,763 rows**, spanning 2013-10-09 14:06:17 to 2015-07-10 11:56:32. The timestamp parse produced zero invalid timestamps. There were **761,551 duplicate timestamp rows**; keeping the last record left **7,757,212 rows**. The median observed interval was **7 seconds**. A total of **3,581,302 intervals** differed from the median by more than the 2-second tolerance, and **1,040,974 intervals** were classified as long gaps. These interval counts describe the timestamp inspection and are not necessarily independent faults.

No negative readings were found in any power column. Readings above the **4,000 W** investigation threshold were flagged, not automatically removed. The flagged counts were: Aggregate **41,054**, Appliance1 **846**, Appliance2 **40**, Appliance3 **118**, Appliance4 **2,085**, Appliance5 **221**, Appliance6 **36**, Appliance7 **374**, Appliance8 **179** and Appliance9 **50**. The maximum observed values ranged from 29,159 W for Aggregate to 98,301 W among the appliance channels.

After one-minute resampling, the series contained **920,031 minute rows**. Before interpolation, missing counts ranged from **115,121** (Appliance7) to **212,969** (Appliance9). I interpolated only complete internal gaps of up to three consecutive missing minutes. Longer gaps and boundary gaps remained missing. After this short-gap fill, **215,381 timestamps** still had at least one missing power channel, and **114,382 timestamps** had all power channels missing. The `Unix` column was excluded from power resampling because it is a timestamp-related field, not a power measurement.

## One-minute washing-machine data quality

The following values are from the generated household quality report. Missing percentages are calculated on the one-minute data. `Issues == 1` is the count/percentage of source rows flagged by the REFIT consistency indicator before one-minute resampling.

| House | One-minute rows | Source rows with `Issues == 1` | Missing aggregate readings | Missing washing-machine readings | Washing-machine missing % |
|---|---:|---:|---:|---:|---:|
| 2 | 889,078 | 28,444 (0.50%) | 218,914 | 219,633 | 24.70% |
| 3 | 885,095 | 408,627 (5.84%) | 100,693 | 136,098 | 15.38% |
| 5 | 933,593 | 425,766 (5.73%) | 87,220 | 123,660 | 13.25% |
| 6 | 831,480 | 34,451 (0.55%) | 126,771 | 127,626 | 15.35% |

### Treatment of the REFIT `Issues` flag

For Houses 2, 3, 5 and 6, source appliance readings were conservatively masked where `Issues == 1`. Aggregate readings were retained. Masked appliance values remain missing (`NaN`), not zero. The flag indicates an aggregate/submeter consistency issue; it does not prove that every individual appliance reading in a flagged row is wrong. This is a conservative quality-control choice and increases the amount of missing appliance data, particularly in Houses 3 and 5.

## Washing-machine usage summary

| House | Off time among valid readings | Detected events | Median event duration | Median event energy |
|---|---:|---:|---:|---:|
| 2 | 94.54% | 391 | 98 min | 0.576 kWh |
| 3 | 94.41% | 948 | 52 min | 0.377 kWh |
| 5 | 92.07% | 1,416 | 28 min | 0.122 kWh |
| 6 | 98.10% | 174 | 76.5 min | 0.490 kWh |

Off-time percentages use valid appliance observations; missing readings are excluded from the denominator. Event counts and cycle statistics depend on the event-detection rule described in RESULTS.md and should not be interpreted as verified physical wash programmes.

## Cleaning and analysis decisions

- Parse timestamps using the source format, validate them, and sort chronologically.
- Inspect duplicate timestamps and sampling-interval irregularity before cleaning.
- Resolve duplicate timestamps by keeping the last record.
- Treat negative power readings as missing, not zero.
- Resample power readings to one-minute means.
- Interpolate only short internal gaps up to three consecutive minutes; preserve longer and boundary gaps as missing.
- Apply the `Issues` mask to appliance channels only; retain aggregate readings.
- Do not silently convert missing appliance readings to zero.
- Use the 5 W / 5-minute / 3-minute-gap event rule for consistent descriptive comparisons.

## Reproduction note

The source dataset is not committed to this repository. Obtain it from the official REFIT source, retain the supplied metadata/readme, place the household files in the local data directories expected by the scripts, and verify the household-specific appliance mappings before running the pipeline. The exact download date was not recorded in the supplied run outputs and is therefore intentionally not stated here.

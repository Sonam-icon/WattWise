
import pandas as pd
from pathlib import Path

files = [
    Path("data/Raw/RAW_House1_Part1.csv"),
    Path("data/Raw/RAW_House1_Part2.csv")
]

for file in files:
    print("\nFILE:", file.name)

    df = pd.read_csv(file, low_memory=False)

    timestamp_col = "Time" if "Time" in df.columns else df.columns[0]

    print("Total rows:", len(df))
    print("Timestamp column:", timestamp_col)

    print("\nFirst 10 original timestamps:")
    print(df[timestamp_col].head(10).to_string(index=False))

    print("\nLast 10 original timestamps:")
    print(df[timestamp_col].tail(10).to_string(index=False))

    parsed = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        dayfirst=True
    )

    print("\nSuccessfully parsed:", parsed.notna().sum())
    print("Invalid timestamps:", parsed.isna().sum())

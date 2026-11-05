
from pathlib import Path
import pandas as pd

BASE = Path("data")

folders = {
    "RAW HOUSE 1": BASE / "Raw",
    "CLEANED DATA": BASE / "Cleaned",
}

def inspect_file(file_path):
    print("\n" + "=" * 75)
    print("FILE:", file_path.name)
    print("=" * 75)

    try:
        df = pd.read_csv(file_path, nrows=5, sep=None, engine="python")

        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.to_string(index=False))
        print("\nData types:")
        print(df.dtypes.to_string())

    except Exception as e:
        print("ERROR:", e)


for label, folder in folders.items():
    print("\n\n", "#" * 30, label, "#" * 30)

    if not folder.exists():
        print("Folder not found:", folder)
        continue

    if label == "RAW HOUSE 1":
        files = sorted(folder.glob("RAW_House1_Part*.csv"))
    else:
        files = sorted(folder.glob("CLEAN_House*.csv"))[:4]

    print("Files found:", len(files))

    for file in files:
        inspect_file(file)

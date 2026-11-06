
"""Download official REFIT archives and extract only requested household CSVs.

Note: REFIT is distributed as .7z archives. This script downloads each archive
in full, then extracts only matching household files. It does not promise
HTTP range/partial archive downloads.
"""
from pathlib import Path
import argparse
import requests
import py7zr
from tqdm import tqdm

URLS = {
    "raw": "https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z",
    "cleaned": "https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z",
}

def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing archive: {destination}")
        return
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=destination.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def extract_households(archive: Path, output_dir: Path, houses: list[int], kind: str = "cleaned") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as z:
        names = z.getnames()
        selected = []
        for n in names:
            fname = Path(n).name.lower()
            if not fname.endswith(".csv"):
                continue
            for h in houses:
                if kind == "raw":
                    if f"raw_house{h}_part" in fname or f"raw_house_{h}_part" in fname or fname in {f"raw_house{h}.csv", f"raw_house_{h}.csv"}:
                        selected.append(n)
                else:
                    if fname in {
                        f"clean_house{h}.csv", f"clean_house_{h}.csv",
                        f"house{h}.csv", f"house_{h}.csv"
                    } or fname.startswith(f"clean_house{h}_") or fname.startswith(f"clean_house_{h}_"):
                        selected.append(n)
        selected = sorted(set(selected))
        if not selected:
            candidates = [Path(n).name for n in names if n.lower().endswith(".csv")]
            raise FileNotFoundError(
                f"No target household CSV found in {archive.name} for houses {houses} ({kind}). "
                f"CSV examples in archive: {candidates[:20]}"
            )
        print("Extracting:", *selected, sep="\n  ")
        z.extract(path=output_dir, targets=selected)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-houses", nargs="+", type=int, default=[1])
    p.add_argument("--clean-houses", nargs="+", type=int, default=[2, 3, 5, 6])
    p.add_argument("--keep-archives", action="store_true")
    args = p.parse_args()
    base = Path(__file__).resolve().parent
    archive_dir = base / "data" / "archives"
    for kind, houses in (("raw", args.raw_houses), ("cleaned", args.clean_houses)):
        archive = archive_dir / f"REFIT_{kind}.7z"
        download(URLS[kind], archive)
        extract_households(archive, base / "data" / kind.capitalize(), houses, kind=kind)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
    print("Dataset extraction complete. Verify filenames and channel mappings before analysis.")

if __name__ == "__main__":
    main()

import argparse
import json
import logging
import subprocess
import zipfile
from pathlib import Path

from data.data_prep import file_hash

ARCHIVES = [
    "CCDB_Export_9_January_2025_through_February_2025.zip",
    "CCDB_Export_10_March_2025_through_April_2025.zip",
    "CCDB_Export_11_May_2025_through_June_2025.zip",
    "CCDB_Export_12_July_2025_through_August_2025.zip",
    "CCDB_Export_13_September_2025_through_October_2025.zip",
    "CCDB_Export_14_November_2025_through_December_2025.zip",
]


def main() -> None:
    """Download the six official 2025 archives without extracting them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/2025"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in ARCHIVES:
        path = args.output / name
        url = f"https://files.consumerfinance.gov/f/documents/{name}"
        if not path.exists():
            logging.info("Downloading %s", name)
            part = path.with_suffix(".part")
            subprocess.run([
                "curl", "--location", "--fail", "--retry", "3", "--silent",
                "--show-error", "--output", str(part), url,
            ], check=True)
            part.replace(path)
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt archive: {path}")
        sources.append({"file": name, "url": url, "bytes": path.stat().st_size, "sha256": file_hash(path)})
        logging.info("Verified %s", name)
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

# SupportIQ

SupportIQ is a customer complaint analysis project built one module at a time. Data preparation is implemented. Model training, search, the data question agent, and the dashboard are still planned.

## Setup

Use Python 3.11. The project virtual environment, Docker, and CI select 3.11. Other Python installations on the laptop are unchanged.

```powershell
uv python install 3.11
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -m pytest -q
```

Without uv, create the environment with `py -3.11 -m venv .venv`, activate it, and use `python -m pip install -r requirements.txt`. Select `.venv/Scripts/python.exe` in VS Code. Outside the activated environment, `python` may point to another version.

## Prepare the 2025 dataset

The sources are the six January–December 2025 ZIP exports in the [official CFPB narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). These are archived snapshots; they may not include later revisions.

```powershell
python -m data.download
python -m data.data_prep data/raw/2025 --output data/processed-2025 --year 2025 --size 30000
python -m pytest -q
```

The downloader requires `curl` on PATH and records URLs and SHA-256 hashes in `sources.json`. It checks ZIP integrity and reuses existing downloads. Failed downloads leave a `.part` file that the next attempt replaces.

Preparation reads 10,000 rows at a time directly from the ZIPs. Cleaned rows go into `cleaned.sqlite`, then a proportional sample is selected using seeded hash ordering. SQLite is a local preparation file, not the planned application database. It uses disk space and extra I/O to avoid keeping a year of narratives in RAM.

Required headers are `Complaint ID`, `Consumer complaint narrative`, and `Product`. Year filtering also requires `Date received`. Available company, response, and timely-response fields are retained. You can also supply a single CSV or single-CSV ZIP instead of a directory.

Rows are rejected in this order:

1. Missing or blank complaint ID.
2. Missing or blank narrative.
3. Missing or blank product.
4. Invalid date, when filtering by year.
5. Date outside the selected year.
6. Duplicate complaint ID.
7. Duplicate normalized narrative, ignoring case.

Each rejected row has exactly one reason. A row missing both ID and narrative counts under missing ID. Duplicates are checked across chunks and files. The first valid ID is remembered even if its text is rejected as a duplicate. Archive filenames are sorted lexicographically, then rows retain source order; this is deterministic, not a preference for the earliest date.

Whitespace is normalized; case and punctuation are preserved. Products with fewer than 100 cleaned rows become `Other` for sampling. Each sampled topic needs at least 10 rows. The 70/15/15 split is rounded per topic, so validation and test sizes can differ slightly.

Outputs are `train.csv`, `val.csv`, `test.csv`, `cleaned.sqlite`, and `manifest.json`. The manifest records source hashes, drop counts, parameters, library versions, split counts, and split hashes. Keep these together and use the same test file for every model comparison. Nonempty output directories are rejected. After a failed run, choose a fresh directory or remove only the incomplete run yourself.

Raw data and generated outputs stay out of Git. The earlier two-month splits remain in `data/processed`; the full-year outputs are in `data/processed-2025`. Sampling changed for chunked processing, so old and new split hashes are not expected to match.

## Validation

The full-year run on November 16, 2026 read all six archives with Python 3.11.15:

| Outcome | Rows |
| --- | ---: |
| Source rows | 5,442,964 |
| Missing narrative | 4,220,914 |
| Duplicate normalized narrative | 614,119 |
| Other drop reasons | 0 |
| Cleaned rows | 607,931 |
| Training sample | 21,001 |
| Validation sample | 4,500 |
| Test sample | 4,499 |

The source total equals dropped plus retained rows. Full source and split hashes, individual drop counts, and elapsed time are in [data_validation_2025.json](reports/data_validation_2025.json).

The earlier two-month run read 811,019 rows and kept 142,633, then sampled 30,000. Its record is in [data_validation.json](reports/data_validation.json). No models have been trained and no model scores are available.

Tests cover cleaning, sequential drop counts, duplicates across chunks/files, rare-class sampling, repeatability, splits, and invalid inputs. When local outputs exist, tests also verify saved hashes, counts, class coverage, stratification, and duplicate IDs/text across splits. Local-data checks skip in CI because complaint data is not uploaded.

## Limits

The cleaned population contains usable narratives after deduplication, not all CFPB complaints. Exact repeated text can come from distinct complaints using the same wording. Removing it helps prevent train/test leakage but changes business counts. Company counts are not failure rates: customer base sizes and submission rates are unavailable. Company response categories and timely responses do not prove resolution.

Deduplication does not catch paraphrases. Random splits do not establish performance on future complaints. Rare-topic mapping uses cleaned label counts to define the dataset before splitting. Learned text preprocessing must use training data only. This module does not add PII masking.

## Docker and CI

The current Dockerfile runs only data preparation:

```powershell
docker build -f deploy/Dockerfile.data -t supportiq-data .
docker run --rm supportiq-data python -m pytest -q
docker run --rm --mount "type=bind,source=$($PWD.Path)/data,target=/app/data" supportiq-data python -m data.data_prep data/raw/2025 --output data/processed-2025 --year 2025
```

The output directory must be empty. Download archives on the host first. GitHub Actions tests with Python 3.11, builds this image, and runs tests inside it. Docker is not installed on the validation laptop.

## Challenges I faced

- [Fill in a problem you encountered, how you investigated it, and what happened.]

## Decisions and trade-offs

- [Fill in a decision you made, alternatives you considered, and the trade-off you accepted.]

## Deployment

Not deployed. Deployment instructions and a demo link will be added after the application is built and tested.

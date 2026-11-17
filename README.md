# SupportIQ

SupportIQ is a customer complaint analysis project built one module at a time. Data preparation and an initial EDA notebook are implemented. Model training, search, the data question agent, and the dashboard are still planned.

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
python -m data.analytics_prep
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

Raw data and generated outputs stay out of Git. The earlier two-month splits remain in `data/processed`; the full-year model outputs are in `data/processed-2025`. Sampling changed for chunked processing, so old and new split hashes are not expected to match.

### Analytics population

`data.analytics_prep` makes a separate set of compressed Parquet files in `data/analytics-2025`. It keeps every source row, including rows without narratives and repeated text. The selected source fields are preserved, with dates normalized to ISO format. Narrative text is not rewritten. An invalid date or date outside 2025 stops this export instead of silently dropping a complaint.

`has_narrative` is false for empty or whitespace-only text. `is_duplicate_text` is true for each later occurrence of the same normalized, case-insensitive narrative; blank narratives are not flagged as duplicates. Flags carry across files in the same sorted archive order used by model preparation. The first occurrence stays unflagged. Identical text is not proof that two records describe the same real-world event.

Use these full-row analytics files for business counts and the deduplicated CSV splits for modeling. Parquet lets the notebook read just the metadata columns it needs without loading complaint text. Preserving rows costs additional disk space; compressed column storage keeps that cost lower than another CSV export.

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

The separate analytics export retains all 5,442,964 source rows, including 1,222,050 with narratives and 614,119 flagged later copies of narrative text. Its file hashes and monthly totals are in [analytics_validation_2025.json](reports/analytics_validation_2025.json). The current suite passed all 21 local tests, including exhaustive verification of the refined sample's saved similarity vectors. Real-data checks are skipped in CI because the data files stay local.

The earlier two-month run read 811,019 rows and kept 142,633, then sampled 30,000. Its record is in [data_validation.json](reports/data_validation.json). No text classifier has been trained yet; the majority-class control below is the first comparison score.

Tests cover cleaning, sequential drop counts, duplicates across chunks/files, rare-class sampling, repeatability, splits, and invalid inputs. When local outputs exist, tests also verify saved hashes, counts, class coverage, stratification, and duplicate IDs/text across splits. Local-data checks skip in CI because complaint data is not uploaded.

## Limits

The model population contains usable narratives after deduplication, not all CFPB complaints. Exact repeated text can come from distinct complaints using the same wording. Model deduplication helps prevent leakage; business counts use all analytics rows instead. Company counts are not failure rates: customer base sizes and submission rates are unavailable. Company response categories and timely responses do not prove resolution.

Deduplication does not catch paraphrases. Random splits do not establish performance on future complaints. Rare-topic mapping uses cleaned label counts to define the dataset before splitting. Learned text preprocessing must use training data only. This module does not add PII masking.

## Split audit and evaluation rules

The active model dataset is now `data/processed-2025-v2`. The earlier `processed-2025` splits are retained for reproducibility and must not be used for new model comparisons. Business analytics still uses every source row.

```powershell
python -m data.refine_splits
python -m ml.majority
```

Refinement refuses to overwrite existing outputs. It checks every pair in the original 30,000-row sample using cosine similarity of L2-normalized word unigram/bigram TF-IDF vectors. It uses scikit-learn's default word tokenization, without stopword removal or vocabulary pruning. Scores strictly above 0.9 are considered near duplicates. This is an exact all-pairs calculation under that lexical definition, not approximate neighbor search or a guarantee about semantic paraphrases. It does not audit every pair in the full 607,931-row model pool.

The audit vocabulary and IDF are fitted once on the original sample and frozen for verification. They are only for dataset hygiene, never model features. Future model vectorizers must be fitted on the training split alone.

| Audit result | Count |
| --- | ---: |
| All near-duplicate pairs, including within splits | 528,065 |
| Cross-split pairs before cleanup | 244,926 |
| Distinct rows in cross-split pairs | 5,112 |
| Rows removed before rebuilding | 4,575 |
| Rows retained | 25,425 |
| Near-duplicate pairs after cleanup | 0 |

Many pairs can involve the same row. A seeded complaint-ID order determines which rows survive: keep an available row and remove its direct near neighbors. This also removes within-split matches so rebuilding cannot move a retained pair across boundaries. It avoids collapsing an entire similarity chain when its endpoints are not similar. A second exhaustive pass verifies the retained vectors. No replacement rows are added, so the audited sample is smaller than the original 30,000.

### Class counts

| Class | Original train | Original val | Original test | Final train | Final val | Final test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Checking or savings account | 1,627 | 349 | 348 | 1,571 | 337 | 336 |
| Credit card | 1,352 | 290 | 290 | 1,345 | 288 | 288 |
| Credit reporting or other personal consumer reports | 12,322 | 2,640 | 2,641 | 9,502 | 2,036 | 2,037 |
| Debt collection | 2,597 | 556 | 557 | 2,519 | 540 | 539 |
| Money transfer, virtual currency, or money service | 1,357 | 291 | 290 | 1,126 | 241 | 241 |
| Mortgage | 484 | 104 | 104 | 484 | 104 | 104 |
| Student loan | 370 | 79 | 79 | 362 | 78 | 77 |
| Vehicle loan or lease | 402 | 86 | 86 | 402 | 86 | 86 |
| Prepaid card | 132 | 28 | 28 | merged | merged | merged |
| Debt or credit management | 89 | 19 | 19 | merged | merged | merged |
| Payday/title/personal/advance loans | 269 | 58 | 57 | merged | merged | merged |
| Other | 0 | 0 | 0 | 487 | 104 | 105 |

Classes with fewer than 50 test examples are merged into `Other`. `Other` must meet the same minimum; if necessary the next smallest class is merged too. The three merged classes above now have 105 test examples together. The final split sizes are 17,798 train, 3,814 validation, and 3,813 test. This trades product detail for a less fragile per-class evaluation. The minimum is a dataset-design rule, not a claim of statistical significance.

### Metrics and weights

- Macro-F1 is the primary comparison and validation-selection metric for every model. Accuracy and per-class results are secondary. `ml.evaluate.score_predictions` fixes this reporting convention and includes all labels with zero for undefined F1.
- Supervised scikit-learn training must use `ml.evaluate.fit_weighted`, which supplies balanced sample weights from training labels only. Do not combine it with an estimator's own `class_weight`, which would double-weight classes.
- For later DistilBERT training, use `ml.evaluate.class_weights` in label-ID order as the loss weights. Zero-shot LLM classification has no training loss to weight; it still uses the same test labels and macro-F1.
- The deliberately unweighted `most_frequent` control always predicts the largest training class. On the audited test set it scores **0.077379 macro-F1** and **53.4225% accuracy**. No text classifier has been trained yet.

Measured scores and training weights are in [majority_baseline.json](reports/split_audit/majority_baseline.json). [audit.json](reports/split_audit/audit.json), class-count CSVs, removed-row IDs, and a compressed pair list document the cleanup. Raw text and audit vectors remain local.

### Findings to keep in context

Of the 1,222,050 narrative-bearing complaints, 614,119 (**50.25%**) were later exact copies after whitespace normalization and ignoring case. Template letters are a plausible explanation, but this has not been verified. All copies stay in business analytics.

Credit reporting accounts for **88.38% of all 5,442,964 source complaints**. That is the business population, not the narrative-only or refined model population. The different populations must not be described as having the same class distribution.

## Local review page

```powershell
python reports/build_preview.py
python -m http.server 8501 --bind 127.0.0.1 --directory preview
```

Open http://localhost:8501 to inspect counts before/after cleanup, class weights, majority scores, and existing EDA charts. It serves only the `preview` directory, not raw complaint files. This is a static review page, not the planned full application. Regenerate it after changing reports.

## EDA and business insights

After both data preparation commands finish:

```powershell
uv pip install -r requirements-eda.txt
python notebooks/run_eda.py
```

The runner uses the active Python environment for the notebook kernel. You can also open [eda.ipynb](notebooks/eda.ipynb) in VS Code and select the project's Python 3.11 interpreter.

The notebook reads metadata columns in batches from the full analytics population. It produces monthly counts, product and company tables, recorded response categories, timely-response rates with explicit denominators, and a model-sample balance check. It exports CSV tables, four PNG charts, and [business_insights.md](reports/eda/business_insights.md). It does not print complaint text or claim to measure resolution, sentiment, or urgency.

The notebook's sample-balance section documents the original pre-audit 30,000-row sample. It is historical; use `reports/split_audit` for the current class counts and benchmark. The business charts remain valid because analytics rows were not removed.

Notebook execution checks source agreement, row totals, twelve-month coverage, and sample proportions. CI tests data code with small fixtures; it does not download the full dataset or execute the real-data notebook.

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

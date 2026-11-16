# 2025 complaint insights

Source: the six official CFPB 2025 archive exports. Analytics keeps all source rows, including repeated narratives and complaints without narratives. This is an archived snapshot, not a market-wide incidence study.

## Measured observations

- The files contain **5,442,964 complaints** across all twelve months of 2025.
- **1,222,050 (22.45%)** contain a narrative. **4,220,914** do not.
- **614,119 (50.25% of narrative-bearing rows)** are later copies of normalized text already encountered. These remain in business counts.
- The largest recorded product category is **Credit reporting or other personal consumer reports**, with **4,810,302 complaints (88.38%)**.
- The highest monthly volume is **2025-10**, with **519,786 complaints**.
- The company with the most recorded complaints is **EQUIFAX, INC.**, with **1,680,536**. This does not establish a higher failure rate.
- **5,418,415** records indicate a timely response and **24,549** indicate a late response. The timely share is **99.55% among 5,442,964 known responses**; **0** records have an unknown or other value.
- The model pool contains **607,931 unique normalized narratives**. The 30,000-row sample differs from its product proportions by at most **0.0018 percentage points**.

## What to investigate next

- Review sub-issues within the largest product category before proposing operational changes. Product counts alone do not identify causes.
- Compare monthly company/product counts and source revisions before interpreting the peak month as a business event.
- Inspect repeated-text patterns separately; repeated wording can represent distinct customers and should not automatically be treated as invalid.
- Obtain customer-base denominators before comparing company complaint rates.

## Limits

Response categories and timely responses are not a verified resolution rate. This dataset does not yet contain sentiment or urgency predictions, so neither metric is reported. Narrative availability and deduplication change the population used by the future classifier. Random model splits do not measure future-period performance.

## Reproduce

Run `python notebooks/run_eda.py` after preparing both the analytics files and model splits. The notebook checks source hashes in the manifests, row totals, twelve-month coverage, and sample proportions. Full file-hash checks are in the data tests. CSV tables and PNG charts are saved beside this report.

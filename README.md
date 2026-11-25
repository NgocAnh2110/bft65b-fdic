# A Big-Data Pipeline for Bank Failure Early Warning

BFT65B – Big Data Analytics in Finance. This project builds a quarterly FDIC bank panel, audits data quality, and evaluates an early-warning rule for a first recorded bank failure within the next four quarters.

## Environment

The project was run on Windows using PowerShell, Python 3.13.7, Temurin Java 17, and PySpark 4.1.1 in local mode with two threads. Python dependencies are pinned in `requirements.txt`.

## Reproduce the analysis

Open PowerShell in the repository root. Install Java 17 and make sure `java -version` succeeds. Create the Python environment and install dependencies:

```powershell
py -3.13 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
$env:PYSPARK_PYTHON = $py
```

In the same terminal, download the full financial panel and the Failures and Institutions tables. The downloader saves one Parquet file per quarter and records completed quarters in a manifest so an interrupted download can resume.

```powershell
& $py .\fdic_download_starter.py --start 1984-03-31 --end 2026-06-30 --out data_raw/fdic
if ($LASTEXITCODE -ne 0) { throw "Check download completeness before continuing" }
```

Run the remaining scripts in dependency order:

```powershell
$steps = @(
  'audit_panel', 'check_p3asset', 'check_panel_keys', 'check_failures',
  'check_failure_coverage', 'benchmark_storage', 'build_financial_panel',
  'audit_bank_exits', 'check_reports_after_failure', 'audit_post_failure_rows',
  'build_labels', 'build_features', 'benchmark_engines', 'validate_development',
  'audit_validation_costs', 'select_threshold', 'spark_quarterly_check',
  'spark_mllib_demo', 'evaluate_final_test'
)
foreach ($step in $steps) {
  & $py ".\$step.py"
  if ($LASTEXITCODE -ne 0) { throw "Failed at $step" }
}
```

Some checks use counts from the audited snapshot. If FDIC revises historical records, inspect discrepancies before changing these checks. The storage benchmark temporarily creates an uncompressed CSV copy of approximately 404 MB.

## Data and validation design

| Table | Audited snapshot |
|---|---:|
| Financials | 1,682,738 bank-quarter rows, 24,064 certificates, 170 quarters from 1984Q1 to 2026Q2 |
| Failures and assistance | 4,117 event rows |
| Institutions | 27,838 rows |

A bank-quarter is positive when the first recorded FAILURE for its certificate occurs after the report date and on or before the end of the fourth following quarter. The modelling sample ends at 2025Q2. We excluded 1,211 observations dated on or after the first recorded failure. Financial features use the exact preceding calendar quarter. This lag approximates information availability because historical publication dates and revisions were not reconstructed.

| Period | Observation quarters | Bank-quarters | Positive rows | Distinct first failures |
|---|---|---:|---:|---:|
| Development | 1984Q1–2007Q4 | 1,228,552 | 8,759 | 2,315 |
| Gap | 2008Q1–2008Q4 | 34,003 | 355 | 163 |
| Test | 2009Q1–2014Q4 | 178,642 | 1,650 | 469 |

Model selection used chronological validation within development, with four full gap quarters per fold. Imputation and preprocessing were fitted on the training period. The baseline ranks banks by negative prior-quarter equity-to-assets. Its alarm threshold was selected before the held-out evaluation. The main scenario assumes a false-alarm penalty equal to 1% of the median positive validation failure cost.

## Results

| Evidence | Result |
|---|---|
| Storage | Quarterly Parquet occupied 248.26 MB versus 404.06 MB for uncompressed CSV. |
| Processing | Median quarterly-aggregation runtime was 0.0522 s in Polars and 0.2968 s in pandas. Local Spark reproduced all 96 quarterly aggregates. |
| Held-out test | The baseline raised 3,047 alarms, including 2,031 false alarms, and detected 403 of 469 failures. Average precision was 0.38146 for the baseline and 0.34117 for logistic regression. |
| MLlib demonstration | A separate development-fold logistic regression achieved Spark areaUnderPR of 0.009676 over 180,160 validation observations. |

The `outputs/` directory contains aggregate audit tables, execution plans, benchmarks and evaluation summaries. Bank-level data and predictions are generated locally.

## Sources and AI disclosure

- Apache Parquet. *Overview*. https://parquet.apache.org/docs/overview/
- Federal Deposit Insurance Corporation. BankFind Suite API and field definitions. https://api.fdic.gov/banks/docs/
- Federal Deposit Insurance Corporation. *BankFind Suite: Bank Structure Changes*. https://banks.data.fdic.gov/bankfind-suite/oscr
- Polars. *Optimizations*. Polars User Guide. https://docs.pola.rs/user-guide/lazy/optimizations/
- Apache Spark. *EXPLAIN*. Spark 4.1.1 SQL Reference. https://spark.apache.org/docs/4.1.1/sql-ref-syntax-qry-explain.html
- scikit-learn. *Common pitfalls and recommended practices*, section “Data leakage.” https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
- scikit-learn. *average_precision_score*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- National Economics University. BFT65B individual report assignment and course-provided `fdic_download_starter.py`.
- BFT65B lecture notebooks and slides. https://github.com/ntlong-NEU/big_data_analysis4Fin
- OpenAI ChatGPT assisted with code drafting, troubleshooting, interpretation and report wording. The numerical results were obtained by running the scripts locally.

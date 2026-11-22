# Healthcare Readmission: SQL + Synthetic ML Data

An end-to-end SQL and Machine Learning pipeline designed to predict 30-day hospital readmissions. It
contains a normalized SQLite schema and a Faker-based generator that creates
longitudinal patient histories and derives a real 30-day readmission label from
subsequent admissions.

The default generator creates **exactly 5,000 rows across the three base
tables**:

| Table | Rows | Purpose |
|---|---:|---|
| `patients` | 1,000 | Demographics and baseline comorbidity burden |
| `admissions` | 1,500 | Encounter details, blood pressure, and the `readmitted_30d` target |
| `lab_results` | 2,500 | Lab values linked to admissions |
| **Base total** | **5,000** | |

The derived `patient_features` table adds **1,000 rows**, one per patient.

## Files

| File | Description |
|---|---|
| [`schema.sql`](schema.sql) | Creates the three tables, constraints, indexes, and the `v_admission_features` view |
| [`generate_data.py`](generate_data.py) | Generates and inserts synthetic data with Faker |
| [`queries.sql`](queries.sql) | Example cohort, audit, and feature-engineering queries |
| [`patient_features.sql`](patient_features.sql) | CTE/LAG-based patient-level feature table |
| [`readmission_features.sql`](readmission_features.sql) | Admission-level modeling table with a usable target |
| [`train_readmission_model.py`](train_readmission_model.py) | Random Forest training, grouped evaluation, and artifact export |
| [`requirements.txt`](requirements.txt) | Python dependency |
| `.gitignore` | Excludes local databases and virtual environments |

## Quick start

Requires Python 3.10+ and SQLite (the Python `sqlite3` module is enough to
generate the database).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py
```

Output:

```text
Created healthcare.db with 5,000 total rows:
  patients       1,000
  admissions     1,500
  lab_results    2,500
Readmission rate: 20.0% (300 positive admissions)
Abnormal lab rate: 42.5%
Verified: foreign keys, one or more labs per admission, and 30-day labels.
```

Explore the data with the SQLite CLI:

```bash
sqlite3 -header -column healthcare.db < patient_features.sql
sqlite3 -header -column healthcare.db < readmission_features.sql
sqlite3 -header -column healthcare.db < queries.sql

python train_readmission_model.py
```

Or from Python:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("healthcare.db")
df = pd.read_sql_query("SELECT * FROM v_admission_features", conn)
print(df.shape)
print(df["readmitted_30d"].value_counts(normalize=True))
```

## Data model

```text
patients 1 ────< admissions 1 ────< lab_results
```

- `patients` has one row per person (`mrn` is unique).
- `admissions` has one row per inpatient encounter and references `patients`.
- `lab_results` has one row per test and references `admissions`.
- `readmitted_30d` is the binary ML target on the **index admission**.
- `readmission_date` records the date of the next admission when the label is
  positive, making the target auditable.
- `admissions` also stores admission-time `systolic_bp` and `diastolic_bp`;
  these support the patient-level average blood-pressure feature.
- `length_of_stay_days` is constrained by SQL to match the admission and
  discharge dates.
- `is_abnormal` is constrained by SQL to match the numeric result and its
  reference interval.

The `v_admission_features` view provides one modeling row per admission,
including age at admission, demographics, encounter attributes, lab counts, and
the abnormal-lab rate.

### Target definition

`readmitted_30d = 1` when the same patient has another admission that begins
within 30 days after the current discharge. It is not assigned independently at
random:

1. Every patient receives an index admission.
2. 300 additional admissions are generated 2–22 days after an index discharge.
3. 200 later, unrelated admissions are generated 45–90 days after discharge.
4. After sorting each patient's encounters, the next admission is used to set
   `readmitted_30d` and `readmission_date`.
5. The generator verifies every label against the next admission and exits with
   an error if any label is inconsistent.

This produces 300 positive admissions out of 1,500 (20.0%). The last encounter
in a patient's history is labeled 0 because there is no subsequent encounter in
the dataset.

## Patient feature table

[`patient_features.sql`](patient_features.sql) builds a reusable
`patient_features` table with one row per patient. It uses:

- a `LAG()` window function over each patient's admissions to find the previous
  admission date;
- a CTE chain to isolate the latest admission and aggregate blood pressure;
- a date difference to calculate `days_since_last_admission`;
- `AVG()` over all admissions for `avg_systolic_bp` and `avg_diastolic_bp`.

The table also includes `latest_admission_date`, `previous_admission_date`,
`total_admissions`, and an average mean arterial pressure.

`days_since_last_admission` is null for the 500 patients with only one
admission. Average blood pressure is calculated from all of a patient's
admissions.

### DATEDIFF() dialect note

The project database is SQLite, which does not implement `DATEDIFF()`. The
checked-in query uses its exact SQLite equivalent:

```sql
CAST(julianday(admission_date) - julianday(previous_admission_date) AS INTEGER)
```

For another database, replace that expression with:

```sql
-- MySQL
DATEDIFF(admission_date, previous_admission_date)

-- SQL Server
DATEDIFF(day, previous_admission_date, admission_date)
```

Both return the number of days between the previous and latest admission.

## Random Forest model

Use [`train_readmission_model.py`](train_readmission_model.py) to build the
modeling table (if needed), train the model, and write evaluation artifacts:

```bash
python train_readmission_model.py
```

The script loads `readmission_features` with `sqlite3` + `pandas`, creates a
patient-grouped holdout using `--test-size` (default 20%), preprocesses numeric
and categorical features, and trains a `RandomForestClassifier`.

### Why not train on `patient_features`?

`patient_features` has one row per patient and is useful for reporting, but it
does not contain a usable supervised target. It describes the patient's latest
admission, and the latest admission has no subsequent encounter in the dataset,
so `readmitted_30d` is always 0. A classifier trained on that table would only
learn to predict "no readmission."

`readmission_features` keeps one row per admission and uses only information
available by the end of that admission:

| Feature group | Examples |
|---|---|
| Prior utilization | `days_since_last_admission`, prior admission count, prior readmission count |
| Patient history | age, comorbidity count, expanding average blood pressure |
| Current encounter | admission type, department, diagnosis, length of stay, discharge disposition, current BP |
| Labs | abnormal-lab rate, creatinine, BNP, HbA1c, hemoglobin, WBC, lactate, eGFR |
| Target | `readmitted_30d` |

The split is grouped by `patient_id`, so no patient appears in both train and
test. The script also runs patient-grouped cross-validation on the training
portion for a more stable estimate.

### Current evaluation

The checked-in synthetic data has 1,500 admissions, 300 positive readmissions,
and 1,000 patients. With seed 42:

```text
Train: 1,209 admissions / 800 patients, 20.5% positive
Test:    291 admissions / 200 patients, 17.9% positive

Training CV:
  ROC-AUC  0.721 +/- 0.017
  PR-AUC   0.370 +/- 0.039
  F1       0.470 +/- 0.008

Holdout at threshold 0.50:
  ROC-AUC 0.626  PR-AUC 0.246  Balanced accuracy 0.652
  Precision 0.245  Recall 0.923  F1 0.387
```

The holdout is small and the synthetic readmission mechanism is weak, so these
numbers are for pipeline demonstration, not evidence of clinical performance.
The high recall / low precision reflects the balanced class weights and the
0.50 threshold. Use the saved probabilities and tune a threshold for the
operational objective; do not treat 0.50 as a clinically validated cutoff.

Because the generator creates at most two encounters per patient, the strongest
features are encounter-history indicators (`is_first_admission`,
`prior_admission_count`, and `days_since_last_admission`). The second encounter
is always the last encounter and therefore has no observed readmission. That is
a synthetic-data artifact, not a clinical finding; use a richer longitudinal
generator or real EHR data before drawing clinical conclusions.

### Saved artifacts

`model_artifacts/` is created by the script:

- `readmission_rf.joblib` — fitted preprocessing + Random Forest pipeline
- `metrics.json` — holdout, CV, confusion matrix, and baseline metrics
- `feature_importance.csv` — transformed feature importances
- `test_predictions.csv` — patient/admission IDs, true label, probability, prediction

Run `python train_readmission_model.py --rebuild-features` after changing the
base tables so the derived modeling table is rebuilt from the current data.

## Reproducibility

`--seed` controls both Faker and Python's random generator. The default seed is
`42`.

Admission dates are relative to today unless `--as-of` is supplied. Use a fixed
`--as-of` when you need identical dates across runs:

```bash
python generate_data.py --seed 42 --as-of 2026-11-22
```

The schema script drops and recreates the three tables, so rerunning
`generate_data.py` resets the target database rather than appending to it. It
also drops the derived `patient_features` table; rebuild that table by running
`patient_features.sql` again.

## Generate a larger dataset

The defaults satisfy the 5,000-row requirement. For a more realistic modeling
dataset, pass larger counts:

```bash
python generate_data.py \
  --patients 5000 \
  --admissions 8000 \
  --labs 20000 \
  --seed 7 \
  --as-of 2026-11-22 \
  --db data/readmission.db
```

Constraints:

- `--admissions` must be at least `--patients` because every patient receives a
  baseline admission.
- `--admissions` must be at most twice `--patients`. This demo generates at
  most two encounters per patient: a baseline and either a readmission or a
  later unrelated encounter.
- `--labs` must be at least `--admissions` because every admission receives at
  least one lab result.

## Suggested ML workflow

1. **Build the cohort** from `v_admission_features` (one row per admission).
2. **Split by patient**, not by row. A random row-level split leaks information
   because the same patient can appear in train and test. Use
   `GroupShuffleSplit(groups=patient_id)` or a time-based split on
   `admission_date`.
3. **Exclude outcome-derived columns** from the feature matrix:
   `readmitted_30d` and `readmission_date`.
4. **Use only index-admission data** for each row. Do not join future
   admissions or post-discharge labs into the features.
5. **Engineer features** from `patients`, `admissions`, and aggregated
   `lab_results`: age, comorbidity count, admission type, length of stay,
   discharge disposition, abnormal-lab rate, and selected lab values.
6. **Handle class imbalance and calibration.** The default dataset has only 300
   positive admissions, so use stratified/group-aware splits, report precision,
   recall, F1, ROC-AUC, and PR-AUC, and consider probability calibration.
7. **Review fairness and clinical validity** before using any model. This
   dataset is synthetic; the relationships are plausible by construction, not
   clinically validated.

The admission-level table is built by `readmission_features.sql`, and
`train_readmission_model.py` trains and evaluates the Random Forest from it. A
minimal encounter-level feature query is also included in `queries.sql`
(query 6). For a larger project, export the modeling table to CSV/Parquet for
scikit-learn, XGBoost, or another library.

## Important limitations

This is synthetic data for software development, SQL practice, and ML pipeline
prototyping. It does **not** represent real patients, real hospitals, or real
clinical outcomes, and it must not be used for clinical decision-making. The
diagnoses, lab distributions, and utilization patterns are simplified for
demonstration.

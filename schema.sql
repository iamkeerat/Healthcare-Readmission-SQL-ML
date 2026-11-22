-- ============================================================================
-- Healthcare Readmission Project - SQLite schema
-- ============================================================================
-- Three-table normalized schema:
--   patients     one row per patient
--   admissions   one row per inpatient encounter; readmitted_30d is the ML target
--   lab_results  one row per lab test associated with an admission
--
-- The script is idempotent: it drops and recreates the objects, so running the
-- data generator again produces a clean database.
-- ============================================================================

PRAGMA foreign_keys = ON;

DROP VIEW IF EXISTS v_admission_features;
DROP TABLE IF EXISTS readmission_features;
DROP TABLE IF EXISTS patient_features;
DROP TABLE IF EXISTS lab_results;
DROP TABLE IF EXISTS admissions;
DROP TABLE IF EXISTS patients;

-- ----------------------------------------------------------------------------
-- Patients
-- ----------------------------------------------------------------------------
CREATE TABLE patients (
    patient_id       INTEGER PRIMARY KEY,
    mrn              TEXT    NOT NULL UNIQUE,
    first_name       TEXT    NOT NULL,
    last_name        TEXT    NOT NULL,
    gender           TEXT    NOT NULL
                             CHECK (gender IN ('Female', 'Male', 'Non-binary', 'Other')),
    date_of_birth    TEXT    NOT NULL
                             CHECK (date(date_of_birth) = date_of_birth),
    race             TEXT    NOT NULL,
    ethnicity        TEXT    NOT NULL,
    insurance_type   TEXT    NOT NULL
                             CHECK (insurance_type IN ('Private', 'Medicare', 'Medicaid',
                                                       'Uninsured', 'Other')),
    comorbidity_count INTEGER NOT NULL DEFAULT 0
                             CHECK (comorbidity_count BETWEEN 0 AND 10),
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ----------------------------------------------------------------------------
-- Admissions
-- ----------------------------------------------------------------------------
-- readmitted_30d = 1 means the patient had another admission that started
-- within 30 days after this discharge. readmission_date records that next
-- admission's date so the label is auditable. The pair is intentionally
-- constrained so a positive label can never point at a missing/out-of-window
-- admission.
-- ----------------------------------------------------------------------------
CREATE TABLE admissions (
    admission_id           INTEGER PRIMARY KEY,
    patient_id             INTEGER NOT NULL
                                   REFERENCES patients(patient_id) ON DELETE CASCADE,
    admission_date         TEXT    NOT NULL
                                   CHECK (date(admission_date) = admission_date),
    discharge_date         TEXT    NOT NULL
                                   CHECK (date(discharge_date) = discharge_date),
    admission_type         TEXT    NOT NULL
                                   CHECK (admission_type IN ('Emergency', 'Urgent',
                                                             'Elective', 'Observation')),
    department             TEXT    NOT NULL,
    primary_diagnosis_code TEXT    NOT NULL,
    primary_diagnosis      TEXT    NOT NULL,
    attending_provider     TEXT    NOT NULL,
    discharge_disposition  TEXT    NOT NULL
                                   CHECK (discharge_disposition IN (
                                       'Home',
                                       'Home with home health',
                                       'Skilled nursing facility',
                                       'Inpatient rehabilitation',
                                       'Hospice',
                                       'Left against medical advice'
                                   )),
    systolic_bp            INTEGER NOT NULL
                                   CHECK (systolic_bp BETWEEN 60 AND 260),
    diastolic_bp           INTEGER NOT NULL
                                   CHECK (diastolic_bp BETWEEN 30 AND 160),
    length_of_stay_days    INTEGER NOT NULL
                                   CHECK (length_of_stay_days >= 0),
    readmitted_30d         INTEGER NOT NULL
                                   CHECK (readmitted_30d IN (0, 1)),
    readmission_date       TEXT,

    CHECK (discharge_date >= admission_date),
    CHECK (systolic_bp > diastolic_bp),
    CHECK (length_of_stay_days =
           CAST(julianday(discharge_date) - julianday(admission_date) AS INTEGER)),
    CHECK (
        (readmitted_30d = 1
         AND readmission_date IS NOT NULL
         AND date(readmission_date) = readmission_date
         AND readmission_date > discharge_date
         AND julianday(readmission_date) - julianday(discharge_date) <= 30)
        OR
        (readmitted_30d = 0 AND readmission_date IS NULL)
    )
);

-- ----------------------------------------------------------------------------
-- Lab results
-- ----------------------------------------------------------------------------
-- is_abnormal is derived from the result and the supplied reference interval.
-- Keeping the interval on each row makes the data useful for feature
-- engineering (for example, "distance outside normal range") without needing a
-- separate reference-range table.
-- ----------------------------------------------------------------------------
CREATE TABLE lab_results (
    lab_result_id   INTEGER PRIMARY KEY,
    admission_id    INTEGER NOT NULL
                            REFERENCES admissions(admission_id) ON DELETE CASCADE,
    test_code       TEXT    NOT NULL,
    test_name       TEXT    NOT NULL,
    result_value    REAL    NOT NULL CHECK (result_value >= 0),
    unit            TEXT    NOT NULL,
    reference_low   REAL    NOT NULL,
    reference_high  REAL    NOT NULL,
    is_abnormal     INTEGER NOT NULL CHECK (is_abnormal IN (0, 1)),
    collected_at    TEXT    NOT NULL
                            CHECK (datetime(collected_at) IS NOT NULL),

    CHECK (reference_low <= reference_high),
    CHECK (
        (result_value < reference_low  AND is_abnormal = 1)
        OR (result_value > reference_high AND is_abnormal = 1)
        OR (result_value BETWEEN reference_low AND reference_high AND is_abnormal = 0)
    )
);

-- ----------------------------------------------------------------------------
-- Query indexes
-- ----------------------------------------------------------------------------
CREATE INDEX idx_admissions_patient_date
    ON admissions(patient_id, admission_date);

CREATE INDEX idx_admissions_discharge
    ON admissions(discharge_date);

CREATE INDEX idx_lab_results_admission
    ON lab_results(admission_id);

CREATE INDEX idx_lab_results_test
    ON lab_results(test_code);

-- ----------------------------------------------------------------------------
-- Convenience view for the readmission modeling cohort
-- ----------------------------------------------------------------------------
CREATE VIEW v_admission_features AS
SELECT
    a.admission_id,
    a.patient_id,
    p.mrn,
    a.admission_date,
    a.discharge_date,
    CAST(strftime('%Y', a.admission_date) AS INTEGER)
        - CAST(strftime('%Y', p.date_of_birth) AS INTEGER)
        - CASE
            WHEN strftime('%m-%d', a.admission_date)
                 < strftime('%m-%d', p.date_of_birth)
            THEN 1 ELSE 0
          END AS age_at_admission,
    p.gender,
    p.race,
    p.ethnicity,
    p.insurance_type,
    p.comorbidity_count,
    a.admission_type,
    a.department,
    a.primary_diagnosis_code,
    a.primary_diagnosis,
    a.discharge_disposition,
    a.systolic_bp,
    a.diastolic_bp,
    a.length_of_stay_days,
    (SELECT COUNT(*)
       FROM lab_results lr
      WHERE lr.admission_id = a.admission_id) AS lab_result_count,
    (SELECT ROUND(AVG(lr.is_abnormal), 3)
       FROM lab_results lr
      WHERE lr.admission_id = a.admission_id) AS abnormal_lab_rate,
    a.readmitted_30d
FROM admissions a
JOIN patients p ON p.patient_id = a.patient_id;

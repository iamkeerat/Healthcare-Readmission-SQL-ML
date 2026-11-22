-- ============================================================================
-- Admission-level readmission modeling table
-- ============================================================================
-- The patient_features table is one row per patient and is useful for
-- reporting, but it cannot train a readmission classifier: the latest
-- admission is right-censored, so every patient_features row has
-- readmitted_30d = 0.
--
-- This table keeps one row per admission and uses only information available
-- by the end of that admission:
--   * days since the previous admission (LAG)
--   * prior admission/readmission counts
--   * patient average blood pressure through the current admission
--   * current admission blood pressure, utilization, and lab aggregates
--   * readmitted_30d, the target
--
-- Rows are ordered by patient and admission date before window functions are
-- applied. No future admission or future lab is used in a feature.
-- ============================================================================

DROP TABLE IF EXISTS readmission_features;

CREATE TABLE readmission_features AS
WITH admission_history AS (
    SELECT
        a.admission_id,
        a.patient_id,
        a.admission_date,
        a.discharge_date,
        a.admission_type,
        a.department,
        a.primary_diagnosis_code,
        a.primary_diagnosis,
        a.discharge_disposition,
        a.length_of_stay_days,
        a.systolic_bp,
        a.diastolic_bp,
        a.readmitted_30d,
        LAG(a.admission_date) OVER (
            PARTITION BY a.patient_id
            ORDER BY a.admission_date, a.admission_id
        ) AS previous_admission_date,
        ROW_NUMBER() OVER (
            PARTITION BY a.patient_id
            ORDER BY a.admission_date, a.admission_id
        ) - 1 AS prior_admission_count,
        COALESCE(
            SUM(a.readmitted_30d) OVER (
                PARTITION BY a.patient_id
                ORDER BY a.admission_date, a.admission_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS prior_readmission_count,
        AVG(a.systolic_bp) OVER (
            PARTITION BY a.patient_id
            ORDER BY a.admission_date, a.admission_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS patient_avg_systolic_bp,
        AVG(a.diastolic_bp) OVER (
            PARTITION BY a.patient_id
            ORDER BY a.admission_date, a.admission_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS patient_avg_diastolic_bp
    FROM admissions a
),
with_gap AS (
    SELECT
        ah.*,
        CASE
            WHEN ah.previous_admission_date IS NULL THEN NULL
            ELSE CAST(
                julianday(ah.admission_date) - julianday(ah.previous_admission_date)
                AS INTEGER
            )
        END AS days_since_last_admission
    FROM admission_history ah
),
lab_features AS (
    SELECT
        admission_id,
        COUNT(*) AS lab_count,
        AVG(is_abnormal) AS abnormal_lab_rate,
        MAX(CASE WHEN test_code = 'CREATININE' THEN result_value END) AS creatinine,
        MAX(CASE WHEN test_code = 'BNP' THEN result_value END) AS bnp,
        MAX(CASE WHEN test_code = 'HBA1C' THEN result_value END) AS hba1c,
        MAX(CASE WHEN test_code = 'HEMOGLOBIN' THEN result_value END) AS hemoglobin,
        MAX(CASE WHEN test_code = 'WBC' THEN result_value END) AS wbc,
        MAX(CASE WHEN test_code = 'LACTATE' THEN result_value END) AS lactate,
        MAX(CASE WHEN test_code = 'EGFR' THEN result_value END) AS egfr
    FROM lab_results
    GROUP BY admission_id
)
SELECT
    w.admission_id,
    w.patient_id,
    CAST(strftime('%Y', w.admission_date) AS INTEGER)
        - CAST(strftime('%Y', p.date_of_birth) AS INTEGER)
        - CASE
            WHEN strftime('%m-%d', w.admission_date)
                 < strftime('%m-%d', p.date_of_birth)
            THEN 1 ELSE 0
          END AS age_at_admission,
    p.gender,
    p.race,
    p.ethnicity,
    p.insurance_type,
    p.comorbidity_count,
    w.admission_date,
    w.discharge_date,
    w.previous_admission_date,
    w.days_since_last_admission,
    w.prior_admission_count,
    w.prior_readmission_count,
    w.admission_type,
    w.department,
    w.primary_diagnosis_code,
    w.primary_diagnosis,
    w.discharge_disposition,
    w.length_of_stay_days,
    w.systolic_bp,
    w.diastolic_bp,
    ROUND(w.patient_avg_systolic_bp, 1) AS patient_avg_systolic_bp,
    ROUND(w.patient_avg_diastolic_bp, 1) AS patient_avg_diastolic_bp,
    COALESCE(lf.lab_count, 0) AS lab_count,
    ROUND(COALESCE(lf.abnormal_lab_rate, 0), 3) AS abnormal_lab_rate,
    lf.creatinine,
    lf.bnp,
    lf.hba1c,
    lf.hemoglobin,
    lf.wbc,
    lf.lactate,
    lf.egfr,
    w.readmitted_30d
FROM with_gap w
JOIN patients p ON p.patient_id = w.patient_id
LEFT JOIN lab_features lf ON lf.admission_id = w.admission_id;

CREATE INDEX idx_readmission_features_patient
    ON readmission_features(patient_id, admission_date);

CREATE INDEX idx_readmission_features_admission
    ON readmission_features(admission_id);

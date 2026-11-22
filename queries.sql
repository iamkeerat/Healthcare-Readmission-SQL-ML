-- ============================================================================
-- Example queries for the healthcare readmission dataset
-- Run with:
--   sqlite3 -header -column healthcare.db < queries.sql
-- ============================================================================

.headers on
.mode column

-- 1. Row counts -------------------------------------------------------------
SELECT 'patients' AS table_name, COUNT(*) AS row_count FROM patients
UNION ALL
SELECT 'admissions', COUNT(*) FROM admissions
UNION ALL
SELECT 'lab_results', COUNT(*) FROM lab_results
ORDER BY table_name;

-- 2. Overall 30-day readmission rate ---------------------------------------
SELECT
    COUNT(*) AS admissions,
    SUM(readmitted_30d) AS readmissions,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct,
    ROUND(AVG(length_of_stay_days), 1) AS avg_length_of_stay_days
FROM admissions;

-- 3. Readmission rate by primary diagnosis ---------------------------------
SELECT
    primary_diagnosis,
    COUNT(*) AS admissions,
    SUM(readmitted_30d) AS readmissions,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct,
    ROUND(AVG(length_of_stay_days), 1) AS avg_los_days
FROM admissions
GROUP BY primary_diagnosis
ORDER BY admissions DESC, readmission_rate_pct DESC;

-- 4. Readmission rate by admission type ------------------------------------
SELECT
    admission_type,
    COUNT(*) AS admissions,
    SUM(readmitted_30d) AS readmissions,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct
FROM admissions
GROUP BY admission_type
ORDER BY admissions DESC;

-- 5. Audit the 30-day label against the next encounter ----------------------
-- This should return 0 mismatches.
WITH ordered AS (
    SELECT
        admission_id,
        patient_id,
        discharge_date,
        LEAD(admission_date) OVER (
            PARTITION BY patient_id ORDER BY admission_date
        ) AS next_admission_date
    FROM admissions
)
SELECT COUNT(*) AS label_mismatches
FROM ordered o
JOIN admissions a USING (admission_id)
WHERE a.readmitted_30d <> CASE
    WHEN o.next_admission_date IS NOT NULL
     AND julianday(o.next_admission_date) - julianday(o.discharge_date)
         BETWEEN 0 AND 30
    THEN 1
    ELSE 0
END;

-- 6. Modeling features: one row per admission -------------------------------
WITH lab_features AS (
    SELECT
        admission_id,
        COUNT(*) AS lab_count,
        ROUND(AVG(is_abnormal), 3) AS abnormal_lab_rate,
        MAX(CASE WHEN test_code = 'CREATININE' THEN result_value END) AS creatinine,
        MAX(CASE WHEN test_code = 'BNP' THEN result_value END) AS bnp,
        MAX(CASE WHEN test_code = 'HBA1C' THEN result_value END) AS hba1c,
        MAX(CASE WHEN test_code = 'HEMOGLOBIN' THEN result_value END) AS hemoglobin,
        MAX(CASE WHEN test_code = 'WBC' THEN result_value END) AS wbc
    FROM lab_results
    GROUP BY admission_id
)
SELECT
    f.admission_id,
    f.age_at_admission,
    f.gender,
    f.insurance_type,
    f.comorbidity_count,
    f.primary_diagnosis,
    f.systolic_bp,
    f.diastolic_bp,
    f.length_of_stay_days,
    lf.lab_count,
    lf.abnormal_lab_rate,
    lf.creatinine,
    lf.bnp,
    lf.hba1c,
    lf.hemoglobin,
    lf.wbc,
    f.readmitted_30d
FROM v_admission_features f
LEFT JOIN lab_features lf USING (admission_id)
ORDER BY f.admission_id
LIMIT 20;

-- 7. Patients with the most encounters -------------------------------------
SELECT
    p.patient_id,
    p.mrn,
    p.gender,
    p.insurance_type,
    p.comorbidity_count,
    COUNT(a.admission_id) AS admissions,
    SUM(a.readmitted_30d) AS readmission_events
FROM patients p
JOIN admissions a ON a.patient_id = p.patient_id
GROUP BY p.patient_id
ORDER BY admissions DESC, readmission_events DESC
LIMIT 15;

-- 8. Monthly encounter volume and readmission rate -------------------------
SELECT
    strftime('%Y-%m', admission_date) AS admission_month,
    COUNT(*) AS admissions,
    SUM(readmitted_30d) AS readmissions,
    ROUND(AVG(readmitted_30d) * 100, 1) AS readmission_rate_pct
FROM admissions
GROUP BY admission_month
ORDER BY admission_month;

-- 9. Referential-integrity check -------------------------------------------
-- This should return no rows.
PRAGMA foreign_key_check;

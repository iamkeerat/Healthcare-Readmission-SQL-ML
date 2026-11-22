-- ============================================================================
-- Patient feature table: one row per patient
-- ============================================================================
-- Builds a reusable feature table from the admission history:
--   * days_since_last_admission: interval between a patient's latest and
--     previous admission, produced with LAG() over the patient's timeline
--   * avg_systolic_bp / avg_diastolic_bp: average admission blood pressure
--     across all of the patient's admissions
--
-- DATEDIFF() does not exist in SQLite. The executable expression below is the
-- exact SQLite equivalent of:
--   MySQL:      DATEDIFF(admission_date, previous_admission_date)
--   SQL Server: DATEDIFF(day, previous_admission_date, admission_date)
-- ============================================================================

DROP TABLE IF EXISTS patient_features;

CREATE TABLE patient_features AS
WITH admission_history AS (
    SELECT
        a.patient_id,
        a.admission_id,
        a.admission_date,
        a.systolic_bp,
        a.diastolic_bp,
        LAG(a.admission_date) OVER (
            PARTITION BY a.patient_id
            ORDER BY a.admission_date, a.admission_id
        ) AS previous_admission_date,
        ROW_NUMBER() OVER (
            PARTITION BY a.patient_id
            ORDER BY a.admission_date DESC, a.admission_id DESC
        ) AS recency_rank
    FROM admissions a
),
admissions_with_gap AS (
    SELECT
        ah.*,
        CASE
            WHEN ah.previous_admission_date IS NULL THEN NULL
            -- SQLite equivalent of DATEDIFF(day, previous, current)
            ELSE CAST(
                julianday(ah.admission_date) - julianday(ah.previous_admission_date)
                AS INTEGER
            )
        END AS days_since_last_admission
    FROM admission_history ah
),
patient_bp AS (
    SELECT
        patient_id,
        ROUND(AVG(systolic_bp), 1) AS avg_systolic_bp,
        ROUND(AVG(diastolic_bp), 1) AS avg_diastolic_bp
    FROM admissions
    GROUP BY patient_id
),
patient_utilization AS (
    SELECT
        patient_id,
        COUNT(*) AS total_admissions
    FROM admissions
    GROUP BY patient_id
),
latest_admission AS (
    SELECT *
    FROM admissions_with_gap
    WHERE recency_rank = 1
)
SELECT
    p.patient_id,
    p.mrn,
    u.total_admissions,
    la.admission_id AS latest_admission_id,
    la.admission_date AS latest_admission_date,
    la.previous_admission_date,
    la.days_since_last_admission,
    bp.avg_systolic_bp,
    bp.avg_diastolic_bp,
    ROUND(
        (bp.avg_systolic_bp + 2.0 * bp.avg_diastolic_bp) / 3.0,
        1
    ) AS avg_mean_arterial_pressure
FROM patients p
JOIN latest_admission la
    ON la.patient_id = p.patient_id
JOIN patient_bp bp
    ON bp.patient_id = p.patient_id
JOIN patient_utilization u
    ON u.patient_id = p.patient_id;

CREATE UNIQUE INDEX idx_patient_features_patient
    ON patient_features(patient_id);

CREATE INDEX idx_patient_features_days
    ON patient_features(days_since_last_admission);

-- Quick preview. Remove this statement if you are running the file as a pure
-- data-building script.
SELECT *
FROM patient_features
ORDER BY patient_id
LIMIT 10;

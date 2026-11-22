#!/usr/bin/env python3
"""Generate a synthetic healthcare readmission dataset with Faker.

Defaults (5,000 total rows):
    1,000 patients
    1,500 admissions
    2,500 lab results

The generator creates longitudinal patient histories so the 30-day readmission
target is derived from actual subsequent admissions rather than assigned at
random. It then inserts the rows into the SQLite schema in ``schema.sql``.

Usage:
    python generate_data.py
    python generate_data.py --db data/readmission.db --seed 7
    python generate_data.py --patients 5000 --admissions 8000 --labs 20000

Requires: Faker>=37
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from faker import Faker

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENTS = 1_000
DEFAULT_ADMISSIONS = 1_500
DEFAULT_LABS = 2_500

# ---------------------------------------------------------------------------
# Clinical reference data
# ---------------------------------------------------------------------------
# ``lab_effects`` are signed offsets applied when the diagnosis is present:
# positive for high-is-bad tests (BNP, lactate, WBC, ...) and negative for
# low-is-bad tests (hemoglobin, eGFR, platelets).
DIAGNOSES: list[dict[str, Any]] = [
    {
        "code": "I50.9",
        "name": "Heart failure, unspecified",
        "department": "Cardiology",
        "weight": 13,
        "severity": 0.64,
        "los_mean": 4.6,
        "los_sd": 2.2,
        "tests": ["BNP", "CREATININE", "BUN", "POTASSIUM", "SODIUM"],
        "lab_effects": {"BNP": 150.0, "CREATININE": 0.12, "SODIUM": -1.6},
    },
    {
        "code": "J44.1",
        "name": "COPD with acute exacerbation",
        "department": "Pulmonology",
        "weight": 10,
        "severity": 0.55,
        "los_mean": 4.2,
        "los_sd": 2.1,
        "tests": ["WBC", "SODIUM", "POTASSIUM", "GLUCOSE"],
        "lab_effects": {"WBC": 1.8, "GLUCOSE": 12.0},
    },
    {
        "code": "J18.9",
        "name": "Pneumonia, unspecified organism",
        "department": "Pulmonology",
        "weight": 12,
        "severity": 0.62,
        "los_mean": 5.0,
        "los_sd": 2.6,
        "tests": ["WBC", "CREATININE", "BUN", "SODIUM", "LACTATE"],
        "lab_effects": {"WBC": 2.8, "LACTATE": 0.45, "SODIUM": -1.0},
    },
    {
        "code": "A41.9",
        "name": "Sepsis, unspecified organism",
        "department": "Critical Care",
        "weight": 8,
        "severity": 0.93,
        "los_mean": 7.1,
        "los_sd": 3.4,
        "tests": ["LACTATE", "WBC", "CREATININE", "PLATELETS", "BUN"],
        "lab_effects": {
            "LACTATE": 1.8,
            "WBC": 3.0,
            "PLATELETS": -70.0,
            "CREATININE": 0.40,
        },
    },
    {
        "code": "N17.9",
        "name": "Acute kidney failure, unspecified",
        "department": "Nephrology",
        "weight": 7,
        "severity": 0.72,
        "los_mean": 5.0,
        "los_sd": 2.4,
        "tests": ["CREATININE", "BUN", "POTASSIUM", "SODIUM", "EGFR"],
        "lab_effects": {
            "CREATININE": 0.85,
            "BUN": 12.0,
            "EGFR": -28.0,
            "POTASSIUM": 0.50,
        },
    },
    {
        "code": "E11.9",
        "name": "Type 2 diabetes mellitus without complications",
        "department": "Endocrinology",
        "weight": 10,
        "severity": 0.38,
        "los_mean": 3.0,
        "los_sd": 1.6,
        "tests": ["GLUCOSE", "HBA1C", "CREATININE", "POTASSIUM"],
        "lab_effects": {"GLUCOSE": 55.0, "HBA1C": 1.5},
    },
    {
        "code": "I48.91",
        "name": "Unspecified atrial fibrillation",
        "department": "Cardiology",
        "weight": 7,
        "severity": 0.48,
        "los_mean": 3.2,
        "los_sd": 1.8,
        "tests": ["POTASSIUM", "CREATININE", "HEMOGLOBIN", "TROPONIN"],
        "lab_effects": {"TROPONIN": 14.0},
    },
    {
        "code": "K92.2",
        "name": "Gastrointestinal hemorrhage, unspecified",
        "department": "Gastroenterology",
        "weight": 6,
        "severity": 0.70,
        "los_mean": 4.1,
        "los_sd": 2.0,
        "tests": ["HEMOGLOBIN", "BUN", "CREATININE", "PLATELETS"],
        "lab_effects": {"HEMOGLOBIN": -2.0, "BUN": 10.0},
    },
    {
        "code": "I63.9",
        "name": "Cerebral infarction, unspecified",
        "department": "Neurology",
        "weight": 5,
        "severity": 0.78,
        "los_mean": 6.0,
        "los_sd": 3.0,
        "tests": ["GLUCOSE", "CREATININE", "SODIUM", "WBC"],
        "lab_effects": {"GLUCOSE": 22.0},
    },
    {
        "code": "L03.90",
        "name": "Cellulitis, unspecified",
        "department": "Infectious Disease",
        "weight": 6,
        "severity": 0.37,
        "los_mean": 3.5,
        "los_sd": 1.8,
        "tests": ["WBC", "GLUCOSE", "CREATININE"],
        "lab_effects": {"WBC": 1.2, "GLUCOSE": 10.0},
    },
    {
        "code": "N39.0",
        "name": "Urinary tract infection, site not specified",
        "department": "Internal Medicine",
        "weight": 8,
        "severity": 0.33,
        "los_mean": 2.8,
        "los_sd": 1.5,
        "tests": ["WBC", "CREATININE", "BUN", "SODIUM"],
        "lab_effects": {"WBC": 1.5},
    },
]

DIAGNOSIS_BY_CODE: dict[str, dict[str, Any]] = {d["code"]: d for d in DIAGNOSES}

# Reference intervals, centers, and severity-driven shifts. ``direction`` is
# +1 when high values are concerning and -1 when low values are concerning.
LAB_SPECS: dict[str, dict[str, Any]] = {
    "GLUCOSE": {
        "name": "Glucose",
        "unit": "mg/dL",
        "low": 70.0,
        "high": 99.0,
        "center": 88.0,
        "spread": 13.0,
        "direction": 1,
        "shift": 26.0,
        "decimals": 0,
        "min": 25.0,
        "max": 500.0,
    },
    "HBA1C": {
        "name": "Hemoglobin A1c",
        "unit": "%",
        "low": 4.0,
        "high": 5.6,
        "center": 5.2,
        "spread": 0.35,
        "direction": 1,
        "shift": 0.7,
        "decimals": 1,
        "min": 3.0,
        "max": 15.0,
    },
    "CREATININE": {
        "name": "Creatinine",
        "unit": "mg/dL",
        "low": 0.6,
        "high": 1.3,
        "center": 0.90,
        "spread": 0.16,
        "direction": 1,
        "shift": 0.50,
        "decimals": 2,
        "min": 0.20,
        "max": 12.0,
    },
    "BUN": {
        "name": "Blood urea nitrogen",
        "unit": "mg/dL",
        "low": 7.0,
        "high": 20.0,
        "center": 14.0,
        "spread": 3.5,
        "direction": 1,
        "shift": 9.0,
        "decimals": 0,
        "min": 2.0,
        "max": 150.0,
    },
    "HEMOGLOBIN": {
        "name": "Hemoglobin",
        "unit": "g/dL",
        "low": 12.0,
        "high": 17.5,
        "center": 14.0,
        "spread": 1.2,
        "direction": -1,
        "shift": 3.0,
        "decimals": 1,
        "min": 4.0,
        "max": 20.0,
    },
    "WBC": {
        "name": "White blood cell count",
        "unit": "K/uL",
        "low": 4.5,
        "high": 11.0,
        "center": 7.2,
        "spread": 1.7,
        "direction": 1,
        "shift": 4.0,
        "decimals": 1,
        "min": 0.5,
        "max": 50.0,
    },
    "POTASSIUM": {
        "name": "Potassium",
        "unit": "mmol/L",
        "low": 3.5,
        "high": 5.1,
        "center": 4.2,
        "spread": 0.33,
        "direction": 1,
        "shift": 0.7,
        "decimals": 1,
        "min": 2.0,
        "max": 7.5,
    },
    "SODIUM": {
        "name": "Sodium",
        "unit": "mmol/L",
        "low": 135.0,
        "high": 145.0,
        "center": 139.0,
        "spread": 2.6,
        "direction": -1,
        "shift": 5.0,
        "decimals": 0,
        "min": 110.0,
        "max": 160.0,
    },
    "LACTATE": {
        "name": "Lactate",
        "unit": "mmol/L",
        "low": 0.5,
        "high": 2.2,
        "center": 1.0,
        "spread": 0.35,
        "direction": 1,
        "shift": 1.3,
        "decimals": 1,
        "min": 0.2,
        "max": 20.0,
    },
    "BNP": {
        "name": "B-type natriuretic peptide",
        "unit": "pg/mL",
        "low": 0.0,
        "high": 100.0,
        "center": 35.0,
        "spread": 20.0,
        "direction": 1,
        "shift": 100.0,
        "decimals": 0,
        "min": 0.0,
        "max": 5000.0,
    },
    "TROPONIN": {
        "name": "High-sensitivity troponin",
        "unit": "ng/L",
        "low": 0.0,
        "high": 14.0,
        "center": 4.0,
        "spread": 3.0,
        "direction": 1,
        "shift": 8.0,
        "decimals": 0,
        "min": 0.0,
        "max": 500.0,
    },
    "EGFR": {
        "name": "Estimated glomerular filtration rate",
        "unit": "mL/min/1.73m2",
        "low": 90.0,
        "high": 200.0,
        "center": 95.0,
        "spread": 12.0,
        "direction": -1,
        "shift": 20.0,
        "decimals": 0,
        "min": 5.0,
        "max": 200.0,
    },
    "PLATELETS": {
        "name": "Platelet count",
        "unit": "K/uL",
        "low": 150.0,
        "high": 450.0,
        "center": 250.0,
        "spread": 50.0,
        "direction": -1,
        "shift": 110.0,
        "decimals": 0,
        "min": 10.0,
        "max": 700.0,
    },
}

GENDER_CHOICES = [("Female", 49), ("Male", 48), ("Non-binary", 2), ("Other", 1)]
RACE_CHOICES = [
    ("White", 58),
    ("Black or African American", 14),
    ("Asian", 7),
    ("American Indian or Alaska Native", 2),
    ("Native Hawaiian or Other Pacific Islander", 1),
    ("Other", 8),
    ("Unknown", 10),
]
ETHNICITY_CHOICES = [
    ("Not Hispanic or Latino", 78),
    ("Hispanic or Latino", 18),
    ("Unknown", 4),
]
INSURANCE_65_PLUS = [
    ("Medicare", 72),
    ("Private", 20),
    ("Medicaid", 5),
    ("Other", 3),
]
INSURANCE_UNDER_65 = [
    ("Private", 46),
    ("Medicaid", 27),
    ("Uninsured", 12),
    ("Medicare", 10),
    ("Other", 5),
]
ADMISSION_TYPES = {
    "index": [("Emergency", 62), ("Urgent", 20), ("Elective", 13), ("Observation", 5)],
    "readmission": [("Emergency", 82), ("Urgent", 14), ("Observation", 4)],
    "later": [("Emergency", 35), ("Urgent", 20), ("Elective", 40), ("Observation", 5)],
}
# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def weighted_pick(rng: random.Random, choices: list[tuple[str, int]]) -> str:
    values, weights = zip(*choices)
    return rng.choices(values, weights=weights, k=1)[0]


def age_on(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def pick_diagnosis(rng: random.Random) -> dict[str, Any]:
    return rng.choices(DIAGNOSES, weights=[d["weight"] for d in DIAGNOSES], k=1)[0]


def choose_disposition(severity: float, rng: random.Random) -> str:
    if severity < 0.35:
        weights = [("Home", 85), ("Home with home health", 10),
                   ("Skilled nursing facility", 3), ("Inpatient rehabilitation", 1),
                   ("Left against medical advice", 1)]
    elif severity < 0.65:
        weights = [("Home", 55), ("Home with home health", 28),
                   ("Skilled nursing facility", 10), ("Inpatient rehabilitation", 5),
                   ("Left against medical advice", 2)]
    elif severity < 0.85:
        weights = [("Home", 25), ("Home with home health", 30),
                   ("Skilled nursing facility", 25), ("Inpatient rehabilitation", 15),
                   ("Hospice", 2), ("Left against medical advice", 3)]
    else:
        weights = [("Home", 10), ("Home with home health", 20),
                   ("Skilled nursing facility", 35), ("Inpatient rehabilitation", 20),
                   ("Hospice", 10), ("Left against medical advice", 5)]
    return weighted_pick(rng, weights)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def build_patient(
    pid: int, rng: random.Random, fake: Faker, as_of: date
) -> dict[str, Any]:
    age = max(18, min(95, round(rng.gauss(63, 16))))

    try:
        birthday = as_of.replace(year=as_of.year - age)
    except ValueError:  # as_of is 29 February and the target year is not leap
        birthday = as_of.replace(year=as_of.year - age, day=28)
    dob = birthday - timedelta(days=rng.randint(0, 364))

    if age >= 65:
        insurance = weighted_pick(rng, INSURANCE_65_PLUS)
    else:
        insurance = weighted_pick(rng, INSURANCE_UNDER_65)

    # Pick the name from the corresponding Faker name pool so the synthetic
    # demographics are internally consistent.
    gender = weighted_pick(rng, GENDER_CHOICES)
    if gender == "Female":
        first_name = fake.first_name_female()
    elif gender == "Male":
        first_name = fake.first_name_male()
    else:
        first_name = fake.first_name()

    # Patients were registered before the 4-year admission observation window.
    created_date = as_of - timedelta(days=rng.randint(365 * 4, 365 * 6))
    created_at = datetime.combine(
        created_date, time(rng.randint(8, 17), rng.randint(0, 59))
    ).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "patient_id": pid,
        "mrn": f"MRN{pid:07d}",
        "first_name": first_name,
        "last_name": fake.last_name(),
        "gender": gender,
        "date_of_birth": dob,
        "race": weighted_pick(rng, RACE_CHOICES),
        "ethnicity": weighted_pick(rng, ETHNICITY_CHOICES),
        "insurance_type": insurance,
        "comorbidity_count": rng.choices(
            range(0, 7), weights=[15, 24, 25, 17, 10, 6, 3], k=1
        )[0],
        "created_at": created_at,
    }


def build_admission(
    patient: dict[str, Any],
    admission_date: date,
    diagnosis: dict[str, Any],
    admission_type: str,
    rng: random.Random,
    providers: list[str],
    *,
    severity_boost: float = 0.0,
) -> dict[str, Any]:
    age = age_on(patient["date_of_birth"], admission_date)
    severity = (
        0.55 * diagnosis["severity"]
        + 0.03 * patient["comorbidity_count"]
        + max(0, age - 50) * 0.0025
        + (0.06 if admission_type == "Emergency" else 0.0)
        + rng.gauss(0, 0.08)
        + severity_boost
    )
    severity = max(0.05, min(0.98, severity))

    los_mean = (
        diagnosis["los_mean"]
        + (severity - 0.5) * 4.0
        + (1.0 if admission_type == "Emergency" else 0.0)
    )
    los = max(1, min(21, round(rng.gauss(los_mean, diagnosis["los_sd"]))))
    discharge_date = admission_date + timedelta(days=los)

    # Admission-time blood pressure. Older and sicker patients tend to run
    # higher, while sepsis and GI bleeding produce relative hypotension.
    bp_shift = {
        "A41.9": (-12.0, -7.0),   # sepsis
        "K92.2": (-8.0, -5.0),    # GI bleeding
        "I63.9": (8.0, 3.0),      # stroke
        "N17.9": (6.0, 2.0),      # acute kidney injury
        "I50.9": (4.0, 1.0),      # heart failure
    }.get(diagnosis["code"], (0.0, 0.0))
    systolic = (
        116.0
        + max(0, age - 50) * 0.35
        + severity * 8.0
        + bp_shift[0]
        + rng.gauss(0, 14)
    )
    diastolic = (
        73.0
        + max(0, age - 50) * 0.10
        + severity * 4.0
        + bp_shift[1]
        + rng.gauss(0, 8)
    )
    systolic = max(80, min(220, round(systolic)))
    diastolic = max(40, min(130, round(diastolic)))
    if systolic <= diastolic + 15:
        systolic = min(240, diastolic + 20)

    return {
        "patient_id": patient["patient_id"],
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "admission_type": admission_type,
        "department": diagnosis["department"],
        "primary_diagnosis_code": diagnosis["code"],
        "primary_diagnosis": diagnosis["name"],
        "attending_provider": rng.choice(providers),
        "discharge_disposition": choose_disposition(severity, rng),
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "length_of_stay_days": los,
        "readmitted_30d": 0,
        "readmission_date": None,
        "_severity": severity,
    }


def build_lab_result(
    admission: dict[str, Any],
    test_code: str,
    lab_result_id: int,
    rng: random.Random,
) -> dict[str, Any]:
    spec = LAB_SPECS[test_code]
    diagnosis = DIAGNOSIS_BY_CODE[admission["primary_diagnosis_code"]]

    # A positive readmission outcome is partly a proxy for underlying severity,
    # so make the index-admission labs modestly worse for those patients.
    severity = admission["_severity"] + (0.06 if admission["readmitted_30d"] else 0.0)
    severity = min(0.99, severity)

    effect = diagnosis["lab_effects"].get(test_code, 0.0)
    value = (
        spec["center"]
        + spec["direction"] * spec["shift"] * severity
        + effect * (0.35 + 0.65 * severity)
        + rng.gauss(0, spec["spread"])
    )
    value = max(spec["min"], min(spec["max"], value))
    value = round(value, spec["decimals"])

    window_start = datetime.combine(admission["admission_date"], time(6, 0))
    window_end = datetime.combine(admission["discharge_date"], time(20, 0))
    if window_end <= window_start:
        window_end = window_start + timedelta(hours=12)
    collected = window_start + timedelta(
        seconds=rng.randint(0, int((window_end - window_start).total_seconds()))
    )

    return {
        "lab_result_id": lab_result_id,
        "admission_id": admission["admission_id"],
        "test_code": test_code,
        "test_name": spec["name"],
        "result_value": value,
        "unit": spec["unit"],
        "reference_low": spec["low"],
        "reference_high": spec["high"],
        "is_abnormal": int(value < spec["low"] or value > spec["high"]),
        "collected_at": collected.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def build_dataset(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(args.seed)
    fake = Faker("en_US")
    fake.seed_instance(args.seed)
    as_of: date = args.as_of

    patients = [build_patient(pid, rng, fake, as_of) for pid in range(1, args.patients + 1)]
    providers = [fake.name() for _ in range(30)]

    admissions: list[dict[str, Any]] = []
    admissions_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def add(patient: dict[str, Any], row: dict[str, Any]) -> None:
        admissions.append(row)
        admissions_by_patient[patient["patient_id"]].append(row)

    # One index admission per patient. The window ends 180 days before the
    # as-of date so even a 21-day stay has a fully observed 30-day outcome.
    base_min = as_of - timedelta(days=365 * 4)
    base_max = as_of - timedelta(days=180)
    for patient in patients:
        start = base_min + timedelta(days=rng.randint(0, (base_max - base_min).days))
        diagnosis = pick_diagnosis(rng)
        admission_type = weighted_pick(rng, ADMISSION_TYPES["index"])
        add(
            patient,
            build_admission(patient, start, diagnosis, admission_type, rng, providers),
        )

    # Add the remaining admissions. A fixed share are true 30-day
    # readmissions; the rest are later, unrelated encounters.
    extra_count = args.admissions - len(patients)
    readmit_target = min(extra_count, round(args.admissions * 0.20))
    readmit_slots = min(readmit_target, len(patients))
    readmit_patients = rng.sample(patients, readmit_slots)
    readmit_ids = {p["patient_id"] for p in readmit_patients}

    for patient in readmit_patients:
        last = admissions_by_patient[patient["patient_id"]][-1]
        readmit_date = last["discharge_date"] + timedelta(days=rng.randint(2, 22))
        if rng.random() < 0.70:
            diagnosis = DIAGNOSIS_BY_CODE[last["primary_diagnosis_code"]]
        else:
            diagnosis = pick_diagnosis(rng)
        admission_type = weighted_pick(rng, ADMISSION_TYPES["readmission"])
        add(
            patient,
            build_admission(
                patient,
                readmit_date,
                diagnosis,
                admission_type,
                rng,
                providers,
                severity_boost=0.08,
            ),
        )

    later_count = extra_count - readmit_slots
    later_candidates = [p for p in patients if p["patient_id"] not in readmit_ids]
    rng.shuffle(later_candidates)
    for i in range(later_count):
        patient = later_candidates.pop() if later_candidates else rng.choice(patients)
        last = admissions_by_patient[patient["patient_id"]][-1]
        later_date = last["discharge_date"] + timedelta(days=rng.randint(45, 90))
        if later_date > as_of - timedelta(days=31):
            later_date = as_of - timedelta(days=31)
        if (later_date - last["discharge_date"]).days <= 30:
            raise ValueError(
                "Not enough date range to add later admissions; reduce --admissions "
                "or move --as-of later."
            )
        diagnosis = (
            DIAGNOSIS_BY_CODE[last["primary_diagnosis_code"]]
            if rng.random() < 0.25
            else pick_diagnosis(rng)
        )
        admission_type = weighted_pick(rng, ADMISSION_TYPES["later"])
        add(
            patient,
            build_admission(
                patient, later_date, diagnosis, admission_type, rng, providers
            ),
        )

    if len(admissions) != args.admissions:
        raise RuntimeError(f"Built {len(admissions)} admissions, expected {args.admissions}")

    # Derive the readmission target from the next admission, per patient.
    for rows in admissions_by_patient.values():
        rows.sort(key=lambda r: (r["admission_date"], r["discharge_date"]))
        for i, current in enumerate(rows):
            current["readmitted_30d"] = 0
            current["readmission_date"] = None
            if i + 1 < len(rows):
                nxt = rows[i + 1]
                gap = (nxt["admission_date"] - current["discharge_date"]).days
                if 0 <= gap <= 30:
                    current["readmitted_30d"] = 1
                    current["readmission_date"] = nxt["admission_date"]

    # Chronological IDs make the database easier to inspect.
    admissions.sort(key=lambda r: (r["patient_id"], r["admission_date"], r["discharge_date"]))
    for admission_id, row in enumerate(admissions, start=1):
        row["admission_id"] = admission_id

    # At least one lab per admission, then random additional tests.
    lab_pairs: list[tuple[dict[str, Any], str]] = []
    for admission in admissions:
        diagnosis = DIAGNOSIS_BY_CODE[admission["primary_diagnosis_code"]]
        lab_pairs.append((admission, rng.choice(diagnosis["tests"])))

    for _ in range(args.labs - len(admissions)):
        admission = rng.choice(admissions)
        diagnosis = DIAGNOSIS_BY_CODE[admission["primary_diagnosis_code"]]
        lab_pairs.append((admission, rng.choice(diagnosis["tests"])))

    labs = [
        build_lab_result(admission, test_code, i, rng)
        for i, (admission, test_code) in enumerate(lab_pairs, start=1)
    ]
    labs.sort(key=lambda r: (r["admission_id"], r["collected_at"]))
    for lab_result_id, row in enumerate(labs, start=1):
        row["lab_result_id"] = lab_result_id

    return patients, admissions, labs


# ---------------------------------------------------------------------------
# Persistence and verification
# ---------------------------------------------------------------------------
def insert_data(
    db_path: Path,
    schema_path: Path,
    patients: list[dict[str, Any]],
    admissions: list[dict[str, Any]],
    labs: list[dict[str, Any]],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema_path.read_text(encoding="utf-8"))

        conn.execute("BEGIN")
        conn.executemany(
            """
            INSERT INTO patients (
                patient_id, mrn, first_name, last_name, gender, date_of_birth,
                race, ethnicity, insurance_type, comorbidity_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["patient_id"],
                    row["mrn"],
                    row["first_name"],
                    row["last_name"],
                    row["gender"],
                    row["date_of_birth"].isoformat(),
                    row["race"],
                    row["ethnicity"],
                    row["insurance_type"],
                    row["comorbidity_count"],
                    row["created_at"],
                )
                for row in patients
            ],
        )

        conn.executemany(
            """
            INSERT INTO admissions (
                admission_id, patient_id, admission_date, discharge_date,
                admission_type, department, primary_diagnosis_code,
                primary_diagnosis, attending_provider, discharge_disposition,
                systolic_bp, diastolic_bp, length_of_stay_days,
                readmitted_30d, readmission_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["admission_id"],
                    row["patient_id"],
                    row["admission_date"].isoformat(),
                    row["discharge_date"].isoformat(),
                    row["admission_type"],
                    row["department"],
                    row["primary_diagnosis_code"],
                    row["primary_diagnosis"],
                    row["attending_provider"],
                    row["discharge_disposition"],
                    row["systolic_bp"],
                    row["diastolic_bp"],
                    row["length_of_stay_days"],
                    row["readmitted_30d"],
                    row["readmission_date"].isoformat()
                    if row["readmission_date"] is not None
                    else None,
                )
                for row in admissions
            ],
        )

        conn.executemany(
            """
            INSERT INTO lab_results (
                lab_result_id, admission_id, test_code, test_name, result_value,
                unit, reference_low, reference_high, is_abnormal, collected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["lab_result_id"],
                    row["admission_id"],
                    row["test_code"],
                    row["test_name"],
                    row["result_value"],
                    row["unit"],
                    row["reference_low"],
                    row["reference_high"],
                    row["is_abnormal"],
                    row["collected_at"],
                )
                for row in labs
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def verify_database(
    db_path: Path, expected_total: int
) -> tuple[dict[str, int], float, float]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("patients", "admissions", "lab_results")
        }
        if sum(counts.values()) != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} total rows, found {sum(counts.values())}"
            )

        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if fk_errors:
            raise RuntimeError(f"Foreign key violations found: {fk_errors}")

        admissions_without_labs = conn.execute(
            """
            SELECT COUNT(*)
            FROM admissions a
            WHERE NOT EXISTS (
                SELECT 1 FROM lab_results lr WHERE lr.admission_id = a.admission_id
            )
            """
        ).fetchone()[0]
        if admissions_without_labs:
            raise RuntimeError(f"{admissions_without_labs} admissions have no lab results")

        label_mismatches = conn.execute(
            """
            SELECT COUNT(*)
            FROM admissions a
            WHERE a.readmitted_30d != CASE WHEN EXISTS (
                SELECT 1
                FROM admissions b
                WHERE b.patient_id = a.patient_id
                  AND b.admission_date > a.discharge_date
                  AND julianday(b.admission_date) - julianday(a.discharge_date) <= 30
            ) THEN 1 ELSE 0 END
            """
        ).fetchone()[0]
        if label_mismatches:
            raise RuntimeError(f"{label_mismatches} admissions have inconsistent labels")

        readmit_rate = conn.execute(
            "SELECT AVG(readmitted_30d) FROM admissions"
        ).fetchone()[0]
        abnormal_rate = conn.execute(
            "SELECT AVG(is_abnormal) FROM lab_results"
        ).fetchone()[0]
        return counts, float(readmit_rate), float(abnormal_rate)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic patients, admissions, and labs with Faker."
    )
    parser.add_argument("--db", type=Path, default=Path("healthcare.db"),
                        help="SQLite database path (default: healthcare.db)")
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("schema.sql"),
                        help="Schema SQL file (default: schema.sql next to this script)")
    parser.add_argument("--patients", type=int, default=DEFAULT_PATIENTS,
                        help=f"Number of patients (default: {DEFAULT_PATIENTS:,})")
    parser.add_argument("--admissions", type=int, default=DEFAULT_ADMISSIONS,
                        help=f"Number of admissions (default: {DEFAULT_ADMISSIONS:,})")
    parser.add_argument("--labs", type=int, default=DEFAULT_LABS,
                        help=f"Number of lab results (default: {DEFAULT_LABS:,})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible output (default: 42)")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(),
                        metavar="YYYY-MM-DD",
                        help="Reference date for the synthetic history (default: today)")
    args = parser.parse_args(argv)

    if args.patients < 1:
        parser.error("--patients must be at least 1")
    if args.admissions < args.patients:
        parser.error("--admissions must be at least as large as --patients")
    if args.admissions > args.patients * 2:
        parser.error(
            "--admissions must be at most twice --patients; this demo generates "
            "at most two encounters per patient"
        )
    if args.labs < args.admissions:
        parser.error("--labs must be at least as large as --admissions")
    if not args.schema.is_file():
        parser.error(f"schema file not found: {args.schema}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_total = args.patients + args.admissions + args.labs

    patients, admissions, labs = build_dataset(args)
    insert_data(args.db, args.schema, patients, admissions, labs)
    counts, readmit_rate, abnormal_rate = verify_database(args.db, expected_total)

    print(f"Created {args.db} with {expected_total:,} total rows:")
    print(f"  patients     {counts['patients']:>7,}")
    print(f"  admissions   {counts['admissions']:>7,}")
    print(f"  lab_results  {counts['lab_results']:>7,}")
    print(
        f"Readmission rate: {readmit_rate * 100:.1f}% "
        f"({sum(a['readmitted_30d'] for a in admissions):,} positive admissions)"
    )
    print(f"Abnormal lab rate: {abnormal_rate * 100:.1f}%")
    print("Verified: foreign keys, one or more labs per admission, and 30-day labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

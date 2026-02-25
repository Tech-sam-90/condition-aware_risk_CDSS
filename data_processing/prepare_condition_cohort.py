from pathlib import Path

import duckdb


RAW_DIR = "/home/ubuntu/condition-aware_risk_CDSS/data/raw"
PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_condition_cohort(connection: duckdb.DuckDBPyConnection):
    query = f"""
    WITH dx AS (
        SELECT
            subject_id,
            hadm_id,
            icd_code,
            icd_version,
            CASE
                WHEN (icd_version = 9 AND icd_code LIKE '250%')
                  OR (icd_version = 10 AND (icd_code LIKE 'E08%' OR icd_code LIKE 'E09%' OR icd_code LIKE 'E10%' OR icd_code LIKE 'E11%' OR icd_code LIKE 'E13%'))
                    THEN 'diabetes'
                WHEN (icd_version = 9 AND icd_code LIKE '585%')
                  OR (icd_version = 10 AND icd_code LIKE 'N18%')
                    THEN 'ckd'
                WHEN (icd_version = 9 AND icd_code LIKE '428%')
                  OR (icd_version = 10 AND icd_code LIKE 'I50%')
                    THEN 'heart_failure'
                WHEN (icd_version = 9 AND (icd_code LIKE '038%' OR icd_code LIKE '99591%' OR icd_code LIKE '99592%' OR icd_code LIKE '78552%'))
                  OR (icd_version = 10 AND (icd_code LIKE 'A40%' OR icd_code LIKE 'A41%' OR icd_code LIKE 'R652%'))
                    THEN 'sepsis'
                ELSE NULL
            END AS condition_label
        FROM read_csv_auto('{RAW_DIR}/diagnoses_icd.csv', header=true)
    ),
    dx_filtered AS (
        SELECT DISTINCT subject_id, hadm_id, condition_label
        FROM dx
        WHERE condition_label IS NOT NULL
    ),
    icu_base AS (
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.intime,
            i.outtime,
            a.admission_type,
            a.hospital_expire_flag,
            p.gender,
            p.anchor_age,
            p.anchor_year_group
        FROM read_csv_auto('{RAW_DIR}/icustays.csv', header=true) i
        LEFT JOIN read_csv_auto('{RAW_DIR}/admissions.csv', header=true) a
          ON i.hadm_id = a.hadm_id
        LEFT JOIN read_csv_auto('{RAW_DIR}/patients.csv', header=true) p
          ON i.subject_id = p.subject_id
    )
    SELECT
        b.subject_id,
        b.hadm_id,
        b.stay_id,
        b.intime,
        b.outtime,
        b.admission_type,
        b.hospital_expire_flag,
        b.gender,
        b.anchor_age,
        b.anchor_year_group,
        d.condition_label
    FROM icu_base b
    INNER JOIN dx_filtered d
      ON b.subject_id = d.subject_id
     AND b.hadm_id = d.hadm_id
    ORDER BY b.subject_id, b.hadm_id, b.stay_id
    """
    return connection.execute(query).df()


def build_condition_model_table(condition_cohort):
    stay_base_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "admission_type",
        "hospital_expire_flag",
        "gender",
        "anchor_age",
        "anchor_year_group",
    ]
    stay_base = condition_cohort[stay_base_cols].drop_duplicates(subset=["stay_id"]).copy()

    condition_flags = (
        condition_cohort.groupby(["stay_id", "condition_label"]).size().unstack(fill_value=0)
        .rename(
            columns={
                "diabetes": "has_diabetes",
                "ckd": "has_ckd",
                "heart_failure": "has_heart_failure",
                "sepsis": "has_sepsis",
            }
        )
        .reset_index()
    )

    for col in ["has_diabetes", "has_ckd", "has_heart_failure", "has_sepsis"]:
        if col not in condition_flags.columns:
            condition_flags[col] = 0

    out = stay_base.merge(condition_flags, on="stay_id", how="left").fillna(0)
    for col in ["has_diabetes", "has_ckd", "has_heart_failure", "has_sepsis"]:
        out[col] = out[col].astype(int)
    return out


if __name__ == "__main__":
    con = duckdb.connect()
    condition_cohort = build_condition_cohort(con)
    condition_model_table = build_condition_model_table(condition_cohort)

    cohort_out = PROCESSED_DIR / "condition_cohort_table.csv"
    model_out = PROCESSED_DIR / "condition_model_table_v1.csv"

    condition_cohort.to_csv(cohort_out, index=False)
    condition_model_table.to_csv(model_out, index=False)

    print(f"Saved: {cohort_out} | shape={condition_cohort.shape}")
    print(f"Saved: {model_out} | shape={condition_model_table.shape}")

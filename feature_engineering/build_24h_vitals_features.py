from pathlib import Path

import duckdb
import pandas as pd


RAW_DIR = "/home/ubuntu/condition-aware_risk_CDSS/data/raw"
PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")


def run_feature_build():
    con = duckdb.connect()

    model_path = PROCESSED_DIR / "condition_model_table_v1.csv"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run data_processing/prepare_condition_cohort.py first."
        )

    condition_model_table = pd.read_csv(model_path)

    stay_time_df = condition_model_table[["stay_id", "intime"]].copy()
    stay_time_df["stay_id"] = pd.to_numeric(stay_time_df["stay_id"], errors="coerce").astype("Int64")
    stay_time_df["intime"] = pd.to_datetime(stay_time_df["intime"], errors="coerce")
    stay_time_df = stay_time_df.dropna(subset=["stay_id", "intime"]).drop_duplicates(subset=["stay_id"])

    con.register("stay_time_df", stay_time_df)

    vitals_24h_query = f"""
    WITH stays AS (
        SELECT CAST(stay_id AS BIGINT) AS stay_id, CAST(intime AS TIMESTAMP) AS intime
        FROM stay_time_df
    ),
    ce AS (
        SELECT
            CAST(stay_id AS BIGINT) AS stay_id,
            CAST(charttime AS TIMESTAMP) AS charttime,
            itemid,
            valuenum
        FROM read_csv_auto('{RAW_DIR}/chartevents.csv.gz', header = true)
        WHERE itemid IN (220045, 220179, 220180, 220181, 220210, 220277, 223761)
          AND valuenum IS NOT NULL
    ),
    ce_24h AS (
        SELECT
            c.stay_id,
            c.charttime,
            c.itemid,
            c.valuenum,
            CASE
                WHEN c.itemid = 220045 THEN 'heart_rate'
                WHEN c.itemid = 220179 THEN 'sbp'
                WHEN c.itemid = 220180 THEN 'dbp'
                WHEN c.itemid = 220181 THEN 'map'
                WHEN c.itemid = 220210 THEN 'resp_rate'
                WHEN c.itemid = 220277 THEN 'spo2'
                WHEN c.itemid = 223761 THEN 'temp_f'
            END AS vital_name
        FROM ce c
        INNER JOIN stays s ON c.stay_id = s.stay_id
        WHERE c.charttime >= s.intime
          AND c.charttime < s.intime + INTERVAL 24 HOUR
    ),
    vital_agg AS (
        SELECT
            stay_id,
            vital_name,
            COUNT(*) AS n,
            AVG(valuenum) AS mean,
            MIN(valuenum) AS min,
            MAX(valuenum) AS max,
            STDDEV_SAMP(valuenum) AS std
        FROM ce_24h
        GROUP BY stay_id, vital_name
    )
    SELECT *
    FROM vital_agg
    ORDER BY stay_id, vital_name
    """

    vitals_24h_long = con.execute(vitals_24h_query).df()
    vitals_24h_wide = vitals_24h_long.pivot_table(
        index="stay_id", columns="vital_name", values=["n", "mean", "min", "max", "std"], aggfunc="first"
    )
    vitals_24h_wide.columns = [f"{vital}_{stat}" for stat, vital in vitals_24h_wide.columns]
    vitals_24h_wide = vitals_24h_wide.reset_index()

    condition_model_with_vitals = condition_model_table.merge(vitals_24h_wide, on="stay_id", how="left")

    out_long = PROCESSED_DIR / "vitals_24h_long.csv"
    out_wide = PROCESSED_DIR / "vitals_24h_features.csv"
    out_model = PROCESSED_DIR / "condition_model_table_v2_with_24h_vitals.csv"

    vitals_24h_long.to_csv(out_long, index=False)
    vitals_24h_wide.to_csv(out_wide, index=False)
    condition_model_with_vitals.to_csv(out_model, index=False)

    print(f"Saved: {out_long} | shape={vitals_24h_long.shape}")
    print(f"Saved: {out_wide} | shape={vitals_24h_wide.shape}")
    print(f"Saved: {out_model} | shape={condition_model_with_vitals.shape}")


if __name__ == "__main__":
    run_feature_build()

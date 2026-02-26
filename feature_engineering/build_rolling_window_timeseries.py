from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


RAW_DIR = "/home/ubuntu/condition-aware_risk_CDSS/data/raw"
PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "sbp",
    220181: "map",
    220210: "resp_rate",
    220277: "spo2",
    223761: "temp_f",
}
VITAL_COLS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_f"]


def pick_primary_condition(row):
    if row.get("has_sepsis", 0) == 1:
        return "sepsis"
    if row.get("has_heart_failure", 0) == 1:
        return "heart_failure"
    if row.get("has_ckd", 0) == 1:
        return "ckd"
    if row.get("has_diabetes", 0) == 1:
        return "diabetes"
    return "other"


def find_first_sustained_sirs_hour(group: pd.DataFrame) -> float:
    criteria = (
        (group["heart_rate"] > 90).astype(int)
        + (group["resp_rate"] > 20).astype(int)
        + ((group["temp_f"] > 100.4) | (group["temp_f"] < 96.8)).astype(int)
    )
    sirs_like = criteria >= 2
    roll5 = sirs_like.rolling(window=5, min_periods=5).sum()
    sustained_idx = np.where(roll5.values == 5)[0]
    if len(sustained_idx) == 0:
        return np.nan
    first_end = int(sustained_idx[0])
    first_start = first_end - 4
    return float(group.iloc[first_start]["hour_from_icu"])


def make_lead_targets(df: pd.DataFrame, leads=(1, 2, 3)) -> pd.DataFrame:
    out = df.copy()
    for lead in leads:
        name = f"target_event_in_{lead}h"
        out[name] = 0
        has_event = out["zero_hour"].notna()
        out.loc[has_event & (out["hour_from_icu"] == (out["zero_hour"] - lead)), name] = 1
    return out


def process_hourly_chunk(hourly_chunk: pd.DataFrame, mortality_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_chunk = hourly_chunk.sort_values(["stay_id", "hour_from_icu"]).copy()

    hourly_chunk[VITAL_COLS] = (
        hourly_chunk.groupby("stay_id")[VITAL_COLS].ffill().bfill()
    )

    for vital in VITAL_COLS:
        for window in [1, 3, 5]:
            col = f"{vital}_mean_{window}h"
            hourly_chunk[col] = (
                hourly_chunk.groupby("stay_id")[vital]
                .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
            )

        slope_col = f"{vital}_slope_5h"
        hourly_chunk[slope_col] = (
            hourly_chunk.groupby("stay_id")[vital]
            .transform(lambda s: s.diff(periods=4) / 4.0)
            .fillna(0.0)
        )

    hourly_chunk["hr_minus_map"] = hourly_chunk["heart_rate"] - hourly_chunk["map"]

    zero_hours = (
        hourly_chunk.groupby("stay_id")
        .apply(find_first_sustained_sirs_hour)
        .reset_index(name="zero_hour")
    )

    rolling_chunk = hourly_chunk.merge(zero_hours, on="stay_id", how="left")
    rolling_chunk = make_lead_targets(rolling_chunk, leads=(1, 2, 3))
    rolling_chunk = rolling_chunk.merge(mortality_map, on="stay_id", how="left")

    return hourly_chunk, rolling_chunk


def main():
    model_path = PROCESSED_DIR / "condition_model_table_v1.csv"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run data_processing/prepare_condition_cohort.py first."
        )

    condition_model = pd.read_csv(model_path)
    condition_model["condition_input"] = condition_model.apply(pick_primary_condition, axis=1)

    stay_time_df = condition_model[["stay_id", "intime", "condition_input"]].copy()
    stay_time_df["stay_id"] = pd.to_numeric(stay_time_df["stay_id"], errors="coerce").astype("Int64")
    stay_time_df["intime"] = pd.to_datetime(stay_time_df["intime"], errors="coerce")
    stay_time_df = stay_time_df.dropna(subset=["stay_id", "intime"]).drop_duplicates(subset=["stay_id"])

    con = duckdb.connect()
    con.register("stay_time_df", stay_time_df)

    itemids = ", ".join(str(x) for x in VITAL_ITEMIDS)
    create_hourly_query = f"""
    CREATE OR REPLACE TEMP TABLE hourly_base AS
    WITH stays AS (
        SELECT CAST(stay_id AS BIGINT) AS stay_id, CAST(intime AS TIMESTAMP) AS intime, condition_input
        FROM stay_time_df
    ),
    ce AS (
        SELECT
            CAST(stay_id AS BIGINT) AS stay_id,
            CAST(charttime AS TIMESTAMP) AS charttime,
            itemid,
            valuenum
        FROM read_csv_auto('{RAW_DIR}/chartevents.csv.gz', header=true)
        WHERE itemid IN ({itemids})
          AND valuenum IS NOT NULL
    ),
    joined AS (
        SELECT
            c.stay_id,
            s.condition_input,
            date_diff('hour', s.intime, c.charttime) AS hour_from_icu,
            c.itemid,
            c.valuenum
        FROM ce c
        INNER JOIN stays s ON c.stay_id = s.stay_id
        WHERE c.charttime >= s.intime
          AND c.charttime < s.intime + INTERVAL 48 HOUR
          AND date_diff('hour', s.intime, c.charttime) BETWEEN 0 AND 47
    )
    SELECT
        stay_id,
        condition_input,
        hour_from_icu,
        AVG(CASE WHEN itemid = 220045 THEN valuenum END) AS heart_rate,
        AVG(CASE WHEN itemid = 220179 THEN valuenum END) AS sbp,
        AVG(CASE WHEN itemid = 220181 THEN valuenum END) AS map,
        AVG(CASE WHEN itemid = 220210 THEN valuenum END) AS resp_rate,
        AVG(CASE WHEN itemid = 220277 THEN valuenum END) AS spo2,
        AVG(CASE WHEN itemid = 223761 THEN valuenum END) AS temp_f
    FROM joined
    GROUP BY stay_id, condition_input, hour_from_icu
    """
    con.execute(create_hourly_query)

    total_rows = con.execute("SELECT COUNT(*) FROM hourly_base").fetchone()[0]
    if total_rows == 0:
        raise RuntimeError("No vital rows extracted for first 48h.")

    stay_ids = [
        row[0]
        for row in con.execute("SELECT DISTINCT stay_id FROM hourly_base ORDER BY stay_id").fetchall()
    ]

    mortality_map = (
        condition_model[["stay_id", "hospital_expire_flag"]]
        .drop_duplicates("stay_id")
        .copy()
    )
    mortality_map["stay_id"] = pd.to_numeric(mortality_map["stay_id"], errors="coerce").astype("Int64")

    out_hourly = PROCESSED_DIR / "hourly_vitals_48h.csv"
    out_rolling = PROCESSED_DIR / "rolling_window_multicondition_timeseries.csv"

    if out_hourly.exists():
        out_hourly.unlink()
    if out_rolling.exists():
        out_rolling.unlink()

    batch_size = 250
    hourly_rows_written = 0
    rolling_rows_written = 0
    label_stats = {"target_event_in_1h": 0, "target_event_in_2h": 0, "target_event_in_3h": 0}

    for i in range(0, len(stay_ids), batch_size):
        batch_ids = stay_ids[i : i + batch_size]
        ids_sql = ",".join(str(int(x)) for x in batch_ids)
        batch_df = con.execute(
            f"""
            SELECT *
            FROM hourly_base
            WHERE stay_id IN ({ids_sql})
            ORDER BY stay_id, hour_from_icu
            """
        ).df()

        hourly_chunk, rolling_chunk = process_hourly_chunk(batch_df, mortality_map)

        hourly_chunk.to_csv(out_hourly, mode="a", header=not out_hourly.exists(), index=False)
        rolling_chunk.to_csv(out_rolling, mode="a", header=not out_rolling.exists(), index=False)

        hourly_rows_written += len(hourly_chunk)
        rolling_rows_written += len(rolling_chunk)
        for key in label_stats:
            label_stats[key] += int(rolling_chunk[key].sum())

        print(
            f"Processed batch {i // batch_size + 1}/{(len(stay_ids) + batch_size - 1) // batch_size}"
            f" | stays={len(batch_ids)} | hourly_rows={hourly_rows_written}"
        )

    print(f"Saved: {out_hourly} | rows={hourly_rows_written}")
    print(f"Saved: {out_rolling} | rows={rolling_rows_written}")
    print(f"Lead-event positives: {label_stats}")


if __name__ == "__main__":
    main()

import logging
from fastf1.logger import set_log_level

set_log_level('ERROR') 
for name in ['fastf1', 'fastf1.core', 'fastf1.req', 'fastf1._api', 'fastf1.logger']:
    logging.getLogger(name).setLevel(logging.ERROR)

import fastf1 as ff1
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

SEASONS = [2021, 2022, 2023, 2024]
ff1.Cache.enable_cache("cache")

def add_telemetry(laps_df):
    rows = []
    for _, lap in laps_df.iterlaps():
        tel = lap.get_car_data()
        if tel is None or tel.empty:
            continue
        rows.append({
            "Driver": lap["Driver"],
            "LapNumber": lap["LapNumber"],
            "SpeedAvg": float(np.nanmean(tel["Speed"])),
            "SpeedMax": float(np.nanmax(tel["Speed"])),
            "ThrottleMean": float(np.nanmean(tel["Throttle"])),
            "BrakePct": float(np.mean((tel["Brake"] > 0).astype(float))),
            "DRS_Use": float(np.mean(tel["DRS"] >= 10))

        })
    return pd.DataFrame(rows)

def add_gap_to_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    g = laps[["Driver","LapNumber","Position","Time","LapStartTime",
              "SpeedST","SpeedFL","SpeedI1","SpeedI2"]].copy()

    # Ensure timedeltas → seconds
    for c in ["Time","LapStartTime"]:
        if c in g.columns and pd.api.types.is_timedelta64_dtype(g[c]):
            g[c] = g[c].dt.total_seconds()

    # Gap at lap finish line (end-of-lap timing)
    g = g.dropna(subset=["Position","Time"])
    g = g.sort_values(["LapNumber","Position"])
    g["GapToAheadAtLine"] = g.groupby("LapNumber")["Time"].diff().fillna(0.0)

    g["AheadDriver"] = g.groupby("LapNumber")["Driver"].shift(1)

    ahead_feats = g[["LapNumber","Driver","SpeedST"]].rename(
        columns={"Driver":"AheadDriver","SpeedST":"Ahead_SpeedST"}
    )
    g = g.merge(ahead_feats, on=["LapNumber","AheadDriver"], how="left")
    g["SpeedST_DiffToAhead"] = g["SpeedST"] - g["Ahead_SpeedST"]

    return g[["Driver","LapNumber","GapToAheadAtLine","SpeedST_DiffToAhead","AheadDriver"]]

def process_race(season:int, round_no:int) -> pd.DataFrame:
    try:
        session = ff1.get_session(season, round_no, "Race")
        session.load()
    except Exception:
        print(f"{season} Round {round_no}: unavailable (skipped)")
        return

    laps = session.laps.copy()
    weather = session.weather_data.copy()
    
    weather["Time_s"] = weather["Time"].dt.total_seconds()

    cols = [
        "Driver","LapTime","LapStartTime","LapNumber","Stint","PitInTime","PitOutTime",
        "Sector1Time","Sector2Time","Sector3Time","Compound","TyreLife","FreshTyre",
        "TrackStatus","Position"
    ]
    df = laps[cols].copy()

    # convert timedeltas to seconds
    for c in ["LapTime","Sector1Time","Sector2Time","Sector3Time","PitInTime","PitOutTime","LapStartTime","Time"]:
        if c in df.columns:
            df[c] = df[c].dt.total_seconds()

    # join nearest weather by time
    df["LapStartTime_s"] = df["LapStartTime"]
    df = pd.merge_asof(
        df.sort_values("LapStartTime_s"),
        weather.sort_values("Time_s"),
        left_on="LapStartTime_s",
        right_on="Time_s",
        direction="nearest"
    )

    df["PitInLap"]  = df.get("PitInTime").notna() if "PitInTime" in df.columns else False
    df["PitOutLap"] = df.get("PitOutTime").notna() if "PitOutTime" in df.columns else False
    df["TrackStatus"] = df.get("TrackStatus", "").fillna("").astype(str)

    df = df.sort_values(["Driver","LapNumber"])
    df["NextPosition"]    = df.groupby("Driver")["Position"].shift(-1)
    df["OvertakeNextLap"] = (df["NextPosition"] < df["Position"]).astype("Int64")

    is_green   = df["TrackStatus"].str.contains("1")
    clean_mask = (~df["PitInLap"]) & (~df["PitOutLap"]) & is_green
    df = df.loc[clean_mask].copy()

    telemetry = add_telemetry(laps)
    df = df.merge(telemetry, on=["Driver","LapNumber"], how="left")

    gap_df = add_gap_to_ahead(session.laps)
    df = df.merge(
        gap_df, on=["Driver","LapNumber"], how="left"
    )

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].fillna("UNKNOWN")

    df["Season"] = season
    df["RoundNumber"] = session.event.RoundNumber
    df["EventName"] = session.event.EventName
    df["Location"] = session.event.Location

    drop_cols = ["PitInLap","PitOutLap","TrackStatus","PitInTime","PitOutTime",
                 "LapStartTime","LapStartTime_s","Time","Time_s"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df.reset_index(drop=True)

def main():
    full = []
    for season in SEASONS:
        try:
            sched = ff1.get_event_schedule(season, include_testing=False)
            # filter out testing / round 0 if present
            sched = sched.loc[sched.index.astype(int) > 0]
            rounds = [(int(r)) for r, row in sched.iterrows()]
        except Exception:
            rounds = [(r) for r in range(1, 26)]
        for round in rounds:
            if round == 0:   # skip preseason testing
                continue
            try:
                race_df = process_race(season, round)
                full.append(race_df)
            except Exception as e:
                print(f"{season} Round {round}: {e}")

    complete_df = pd.concat(full, ignore_index=True)

    complete_df.to_parquet("../f1.parquet")
    complete_df.to_csv("../f1.csv")

if __name__ == "__main__":
    main()

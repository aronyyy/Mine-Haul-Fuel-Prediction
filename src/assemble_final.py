"""
Blocks 5 & 7 (assembly step) — Final feature assembly.

Merges shift features, trip features, total_trip sensor column,
fuel consumption labels (train only), and computes interaction
features.  For test, also merges the submission ID mapping.

Outputs (written to data/processed/):
  train_final.parquet
  test_final.parquet
"""

import gc
import warnings

import numpy as np
import pandas as pd

from config import OUT_DIR, RAW_DIR, load_fuel_labels

warnings.filterwarnings("ignore")

KEY = ["dumper_id", "date_dpr", "shift"]


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived interaction features shared by train and test."""
    eps    = 1e-6
    ref_d  = df["total_dist_hav_km"].replace(0, np.nan)
    ld_hav = df.get("loaded_dist_hav_km", pd.Series(dtype=float))
    em_hav = df.get("empty_dist_hav_km",  pd.Series(dtype=float))
    spd_l  = df.get("avg_speed_loaded", pd.Series(0, index=df.index))
    spd_e  = df.get("avg_speed_empty",  pd.Series(eps, index=df.index))
    ld_gr  = df.get("mean_uphill_grad", pd.Series(eps, index=df.index))
    tph    = df.get("trips_per_hour",   pd.Series(0, index=df.index))

    df["gradient_work_hav"]   = (df["mean_gradient_pct"] * df["total_dist_hav_km"]).round(4)
    df["gradient_work_spd"]   = (df["mean_gradient_pct"] * df["total_dist_spd_km"]).round(4)
    df["loaded_lift_proxy"]   = (ld_hav * df["net_lift_m"]).fillna(0).round(4)
    df["dist_per_trip"]       = (ref_d  / df["calc_trips"].replace(0, np.nan)).fillna(0).round(4)
    df["loaded_km_per_trip"]  = (ld_hav / df["calc_trips"].replace(0, np.nan)).fillna(0).round(4)
    df["haul_imbalance"]      = (ld_hav - em_hav).abs().fillna(0).round(4)
    df["elev_per_trip"]       = (df["gross_elev_m"] / df["calc_trips"].replace(0, np.nan)).fillna(0).round(4)
    df["speed_efficiency"]    = (spd_l / (spd_e + eps)).round(4)
    df["loaded_kmh_per_grad"] = (spd_l / (ld_gr.abs() + eps)).round(4)
    df["cycle_efficiency"]    = (tph * ld_hav.fillna(0)).fillna(0).round(4)
    df["stoppage_density"]    = df["stopgo_intensity"]   # alias
    return df


def assemble_train() -> None:
    """Block 5 — build train_final.parquet."""
    print("=" * 70)
    print("TRAIN FINAL ASSEMBLY")
    print("=" * 70)

    train = pd.read_parquet(OUT_DIR + "train_features.parquet")
    train["date_dpr"] = pd.to_datetime(train["date_dpr"])

    # Merge trip features
    trip = pd.read_parquet(OUT_DIR + "train_trip_features.parquet")
    trip["date_dpr"] = pd.to_datetime(trip["date_dpr"])
    train = train.merge(trip, on=KEY, how="left")
    del trip
    gc.collect()

    # total_trip sensor column from raw preproc
    print("  Extracting total_trip …")
    raw_tt = pd.read_parquet(
        OUT_DIR + "train_dump_preproc.parquet",
        columns=["vehicle", "date_dpr", "shift_dpr", "total_trip"],
    )
    tt = (
        raw_tt.groupby(["vehicle", "date_dpr", "shift_dpr"], observed=True)["total_trip"]
        .first().reset_index()
        .rename(columns={"vehicle": "dumper_id", "shift_dpr": "shift"})
    )
    tt["date_dpr"] = pd.to_datetime(tt["date_dpr"])
    del raw_tt
    gc.collect()
    if "total_trip" in train.columns:
        train.drop(columns=["total_trip"], inplace=True)
    train = train.merge(tt, on=KEY, how="left")
    del tt
    gc.collect()

    # Fuel label
    print("  Merging fuel label …")
    fuel = load_fuel_labels()
    fuel["had_refill"] = (
        pd.concat([pd.read_csv(f) for f in [
            RAW_DIR + "smry_jan_train_ordered.csv",
            RAW_DIR + "smry_feb_train_ordered.csv",
            RAW_DIR + "smry_mar_train_ordered.csv",
        ]], ignore_index=True)
        .rename(columns=str.lower)
        .pipe(lambda d: (d["arefill"] > 0).astype(int))
        if False else fuel.get("had_refill", pd.Series(dtype=int))
    )
    # Re-load with extra columns from summary CSVs
    parts = []
    for f in [
        RAW_DIR + "smry_jan_train_ordered.csv",
        RAW_DIR + "smry_feb_train_ordered.csv",
        RAW_DIR + "smry_mar_train_ordered.csv",
    ]:
        try:
            parts.append(pd.read_csv(f))
        except FileNotFoundError:
            print(f"  ⚠️  {f} not found")
    fuel_full = pd.concat(parts, ignore_index=True)
    del parts
    fuel_full.columns = fuel_full.columns.str.strip().str.lower()
    fuel_full["fuel_consumed_L"] = (
        fuel_full["initlev"] + fuel_full["arefill"] - fuel_full["endlev"]
    ).clip(lower=0).round(4)
    fuel_full["had_refill"] = (fuel_full["arefill"] > 0).astype(int)
    fuel_full["lph"] = fuel_full["lph"].replace([np.inf, -np.inf], np.nan)
    fuel_full = fuel_full.rename(
        columns={"vehicle": "dumper_id", "date": "date_dpr", "shift": "shift"}
    )
    fuel_full["date_dpr"] = pd.to_datetime(fuel_full["date_dpr"])
    keep = ["dumper_id", "date_dpr", "shift",
            "fuel_consumed_L", "had_refill", "runhrs", "lph", "initlev", "endlev"]
    fuel_full = fuel_full[[c for c in keep if c in fuel_full.columns]]
    train = train.merge(fuel_full, on=KEY, how="left")
    del fuel_full
    gc.collect()

    # Interaction features
    print("  Computing interaction features …")
    train = _add_interaction_features(train)

    n_fuel = train["fuel_consumed_L"].notna().sum()
    print(f"\n  Shape     : {train.shape}")
    print(f"  Fuel rows : {n_fuel:,} / {len(train):,} ({100*n_fuel/len(train):.1f}%)")

    train.to_parquet(OUT_DIR + "train_final.parquet", index=False)
    print(f"  ✅  Saved → train_final.parquet  {train.shape}")
    del train
    gc.collect()
    print("\n  TRAIN FINAL ASSEMBLY COMPLETE ✅")


def assemble_test() -> None:
    """Block 7 (assembly step) — build test_final.parquet."""
    print("=" * 70)
    print("TEST FINAL ASSEMBLY")
    print("=" * 70)

    test = pd.read_parquet(OUT_DIR + "test_features.parquet")
    test["date_dpr"] = pd.to_datetime(test["date_dpr"])

    trip = pd.read_parquet(OUT_DIR + "test_trip_features.parquet")
    trip["date_dpr"] = pd.to_datetime(trip["date_dpr"])
    test = test.merge(trip, on=KEY, how="left")
    del trip
    gc.collect()

    # total_trip (if present in test preproc)
    test_preproc_cols = pd.read_parquet(
        OUT_DIR + "test_dump_preproc.parquet", columns=["total_trip"]
    ).columns.tolist()
    if "total_trip" in test_preproc_cols:
        raw_tt = pd.read_parquet(
            OUT_DIR + "test_dump_preproc.parquet",
            columns=["vehicle", "date_dpr", "shift_dpr", "total_trip"],
        )
        tt = (
            raw_tt.groupby(["vehicle", "date_dpr", "shift_dpr"], observed=True)["total_trip"]
            .first().reset_index()
            .rename(columns={"vehicle": "dumper_id", "shift_dpr": "shift"})
        )
        tt["date_dpr"] = pd.to_datetime(tt["date_dpr"])
        del raw_tt
        gc.collect()
        if "total_trip" in test.columns:
            test.drop(columns=["total_trip"], inplace=True)
        test = test.merge(tt, on=KEY, how="left")
        del tt
        gc.collect()

    # Interaction features
    test = _add_interaction_features(test)

    # Submission ID mapping
    try:
        id_map = pd.read_csv(RAW_DIR + "id_mapping_new.csv")
        id_map["date"] = pd.to_datetime(id_map["date"])
        id_map = id_map.rename(
            columns={"vehicle": "dumper_id", "date": "date_dpr", "shift": "shift"}
        )
        test = test.merge(id_map, on=KEY, how="left")
        print(f"  ID mapped: {test['id'].notna().sum():,} / {len(test):,}")
    except FileNotFoundError:
        print("  ⚠️  id_mapping_new.csv not found")

    miss = test.isnull().sum()
    miss = miss[miss > 0]
    print(f"\n  Test shape: {test.shape}")
    print("  Missing values:")
    print(miss.to_string() if len(miss) else "  None ✅")

    test.to_parquet(OUT_DIR + "test_final.parquet", index=False)
    print(f"  ✅  Saved → test_final.parquet  {test.shape}")
    del test
    gc.collect()
    print("\n  TEST FINAL ASSEMBLY COMPLETE ✅")


if __name__ == "__main__":
    assemble_train()
    assemble_test()

"""
Blocks 3 & 6 — Shift-level feature engineering.

Reads preprocessed dump parquets and aggregates to one row per
(vehicle, date_dpr, shift_dpr).  Both speed×time and haversine
distance variants are kept.  Includes gradient, angle, stoppage,
and GPS quality features.  Time metrics are capped at 8 h.

Outputs (written to data/processed/):
  train_features.parquet
  test_features.parquet
"""

import gc
import warnings

import numpy as np
import pandas as pd

from config import (
    DPR_COLS, OUT_DIR, SHIFT_CAP_HRS, safe_div,
)

warnings.filterwarnings("ignore")

KEY = ["vehicle", "date_dpr", "shift_dpr"]


def build_shift_features(preproc_path: str, output_path: str, is_train: bool = True) -> None:
    """
    Aggregate raw preproc parquet → one row per (vehicle, date_dpr, shift_dpr).

    Both distance variants are computed:
      _dist_spd  = speed × dt_sec / 3600  (km, GPS-ok + moving rows)
      _dist_hav  = disthav / 1000          (km, GPS-ok rows with valid disthav)

    Parameters
    ----------
    preproc_path : str  Path to *_dump_preproc.parquet.
    output_path  : str  Destination for *_features.parquet.
    is_train     : bool Whether to include DPR sanity checks.
    """
    print(f"\n  Loading {preproc_path.split('/')[-1]} …")
    df = pd.read_parquet(preproc_path)
    df = df.sort_values(["vehicle", "ts"]).reset_index(drop=True)
    print(f"  Shape: {df.shape}")

    # ── Row-level derived arrays ──────────────────────────────────────────────
    gap_flag  = df["is_gap_300s"].values == 1.0
    is_blind  = (df["gap_status"] == "engine_running_blind").values if is_train else np.zeros(len(df), bool)
    is_offst  = (df["gap_status"] == "engine_off_stationary").values if is_train else np.zeros(len(df), bool)
    is_normal = ~gap_flag

    dt_eff = np.where(gap_flag, df["actual_dt_sec"].values, df["dt_sec"].values).astype(np.float32)

    ign = df["ignition"].values == 1
    spd = df["speed"].values
    gps = df["gps_ok"].astype(bool).values

    # Time budget
    df["_dt_eon"]   = np.where((is_normal & ign) | is_blind,                     dt_eff, 0).astype(np.float32)
    df["_dt_eoff"]  = np.where((is_normal & ~ign) | is_offst,                    dt_eff, 0).astype(np.float32)
    df["_dt_prod"]  = np.where((is_normal & ign & (spd >= 2)) | is_blind,        dt_eff, 0).astype(np.float32)
    df["_dt_idle"]  = np.where(is_normal & ign & (spd < 2),                      dt_eff, 0).astype(np.float32)
    df["_dt_maint"] = np.where(((is_normal & ~ign) | is_offst) & (dt_eff > 900), dt_eff, 0).astype(np.float32)
    df["_dt_blind"] = np.where(is_blind,                                          dt_eff, 0).astype(np.float32)

    # Distance (dual variants)
    moving_gps   = is_normal & ign & (spd >= 2) & gps
    df["_d_spd"] = np.where(moving_gps, (spd * df["dt_sec"].values) / 3600, 0).astype(np.float32)

    has_dh = "disthav" in df.columns
    if has_dh:
        dh = df["disthav"].fillna(0).values
        df["_d_hav"] = np.where(gps & (dh > 0), dh / 1000, 0).astype(np.float32)
    else:
        df["_d_hav"] = df["_d_spd"].copy()

    has_cumdist   = "cumdist"    in df.columns
    has_totaltrip = "total_trip" in df.columns
    has_rain      = "rain_loss"  in df.columns
    has_fog       = "dense_fog"  in df.columns

    # Altitude diff (zeroed at gap boundaries)
    adiff = df.groupby("vehicle")["altitude"].diff().fillna(0).values.astype(np.float32)
    adiff[gap_flag] = 0.0
    df["_lift"]  = np.where(gps & (adiff > 0),  adiff, 0).astype(np.float32)
    df["_desc"]  = np.where(gps & (adiff < 0), -adiff, 0).astype(np.float32)
    df["_gelev"] = np.where(gps, np.abs(adiff), 0).astype(np.float32)

    # Gradient
    dist_m  = np.where(has_dh, df["_d_hav"].values * 1000, (spd * df["dt_sec"].values) / 3.6)
    valid_g = moving_gps & (dist_m > 5)
    grad    = np.clip(
        np.where(valid_g, adiff / np.where(valid_g, dist_m, np.nan) * 100, np.nan),
        -50, 50,
    )
    df["_grad"]        = grad.astype(np.float32)
    df["_gradabs"]     = np.abs(grad).astype(np.float32)
    df["_grad_up"]     = np.where(valid_g & (grad > 0),  grad,  np.nan).astype(np.float32)
    df["_grad_dn"]     = np.where(valid_g & (grad < 0), -grad,  np.nan).astype(np.float32)
    df["_d_steep_spd"] = np.where(valid_g & (grad > 5), df["_d_spd"].values, 0).astype(np.float32)
    df["_d_steep_hav"] = np.where(valid_g & (grad > 5), df["_d_hav"].values, 0).astype(np.float32)
    df["_d_up_hav"]    = np.where(valid_g & (grad > 0), df["_d_hav"].values, 0).astype(np.float32)
    df["_d_dn_hav"]    = np.where(valid_g & (grad < 0), df["_d_hav"].values, 0).astype(np.float32)

    # Angle delta
    prev_ang   = df.groupby("vehicle")["angle"].shift(1)
    raw_ad     = (df["angle"] - prev_ang).abs()
    angd       = np.where(raw_ad > 180, 360 - raw_ad, raw_ad)
    df["_angd"]       = np.where(moving_gps & prev_ang.notna(), angd, np.nan).astype(np.float32)
    df["_sharp_turn"] = np.where(df["_angd"] > 45, 1, 0).astype(np.int8)

    # Event flags
    prev_ign = df.groupby("vehicle")["ignition"].shift(1).fillna(0)
    prev_spd = df.groupby("vehicle")["speed"].shift(1).fillna(0)
    df["_igns"]   = ((df["ignition"] == 1) & (prev_ign == 0)).astype(np.int8)
    df["_stop"]   = ((spd < 2) & (prev_spd >= 2)).astype(np.int8)
    df["_ngap"]   = gap_flag.astype(np.int8)
    df["_nblind"] = is_blind.astype(np.int8)
    df["_noffst"] = is_offst.astype(np.int8)

    gc.collect()
    print("  ✅  Row-level features computed.")

    # Drop columns not needed for aggregation
    _drop = [c for c in df.columns if c in [
        "latitude", "longitude", "utm_x", "utm_y", "gnss_pdop", "gnss_hdop",
        "satellites", "gsm_operator", "fuel_volume", "fuel_delta",
        "gap_status", "actual_dt_sec", "dt_sec", "is_gap_300s",
        "received_ts", "battery_level", "battery_current", "battery_voltage",
        "external_voltage", "axis_x", "axis_y", "axis_z", "gsm_signal",
    ]]
    df.drop(columns=_drop, inplace=True)
    gc.collect()

    print("  Aggregating per (vehicle, date_dpr, shift_dpr) …")

    # C1. Time + counts + metadata
    a1 = df.groupby(KEY, observed=True).agg(
        engine_on_s    = ("_dt_eon",    "sum"),
        engine_off_s   = ("_dt_eoff",   "sum"),
        prod_s         = ("_dt_prod",   "sum"),
        idle_s         = ("_dt_idle",   "sum"),
        maint_s        = ("_dt_maint",  "sum"),
        blind_s        = ("_dt_blind",  "sum"),
        n_ign_cycles   = ("_igns",      "sum"),
        n_stops        = ("_stop",      "sum"),
        n_gaps         = ("_ngap",      "sum"),
        n_blind_gaps   = ("_nblind",    "sum"),
        n_off_stat     = ("_noffst",    "sum"),
        n_sharp_turns  = ("_sharp_turn","sum"),
        row_count      = ("ts",         "count"),
        gps_ok_pct     = ("gps_ok",     "mean"),
        mine_id        = ("mine_anon",  "first"),
        operator_id    = ("operator_id","first"),
    ).reset_index()

    # C2. Distance
    a2 = df.groupby(KEY, observed=True).agg(
        total_dist_spd_km    = ("_d_spd",       "sum"),
        total_dist_hav_km    = ("_d_hav",       "sum"),
        dist_steep_spd_km    = ("_d_steep_spd", "sum"),
        dist_steep_hav_km    = ("_d_steep_hav", "sum"),
        uphill_dist_hav_km   = ("_d_up_hav",    "sum"),
        downhill_dist_hav_km = ("_d_dn_hav",    "sum"),
    ).reset_index()

    # C3. Elevation
    a3 = df.groupby(KEY, observed=True).agg(
        net_lift_m    = ("_lift",  "sum"),
        net_descent_m = ("_desc",  "sum"),
        gross_elev_m  = ("_gelev", "sum"),
    ).reset_index()

    # C4. Gradient (mean / max / std + directional)
    a4 = df.groupby(KEY, observed=True).agg(
        mean_gradient_pct = ("_grad",    "mean"),
        max_gradient_pct  = ("_gradabs", "max"),
        std_grad          = ("_grad",    "std"),
    ).reset_index()
    a4_up = (
        df[df["_grad_up"].notna()]
        .groupby(KEY, observed=True)["_grad_up"]
        .mean().reset_index()
        .rename(columns={"_grad_up": "mean_uphill_grad"})
    )
    a4_dn = (
        df[df["_grad_dn"].notna()]
        .groupby(KEY, observed=True)["_grad_dn"]
        .mean().reset_index()
        .rename(columns={"_grad_dn": "mean_downhill_grad"})
    )

    # C5. Angle
    a5 = df.groupby(KEY, observed=True).agg(
        mean_angle_change = ("_angd", "mean"),
        std_angle_change  = ("_angd", "std"),
        total_turning_deg = ("_angd", "sum"),
    ).reset_index()

    # C6. Speed (GPS-ok + moving rows only)
    mv_m = df["gps_ok"].astype(bool) & (df["speed"] >= 2) & (df["ignition"] == 1)
    a6   = (
        df[mv_m].groupby(KEY, observed=True)["speed"]
        .agg(mean_speed="mean", max_speed="max", std_speed="std")
        .reset_index()
    )

    # C7. Altitude std (GPS-ok rows)
    a7 = (
        df[df["gps_ok"].astype(bool)]
        .groupby(KEY, observed=True)["altitude"]
        .std().reset_index()
        .rename(columns={"altitude": "alt_std"})
    )

    # Optional sensor columns
    opt = []
    if has_cumdist:
        ca = df.groupby(KEY, observed=True)["cumdist"].agg(_mn="min", _mx="max").reset_index()
        ca["cumdist_range"] = (ca["_mx"] - ca["_mn"]).clip(lower=0)
        ca.drop(columns=["_mn", "_mx"], inplace=True)
        opt.append(ca)
    if has_totaltrip:
        opt.append(df.groupby(KEY, observed=True)["total_trip"].first().reset_index())
    if has_rain:
        opt.append(df.groupby(KEY, observed=True)["rain_loss"].first().reset_index())
    if has_fog:
        opt.append(df.groupby(KEY, observed=True)["dense_fog"].first().reset_index())

    # DPR ground-truth columns (train only)
    avail_dpr = [c for c in DPR_COLS if c in df.columns] if is_train else []
    dpr_agg   = df.groupby(KEY, observed=True)[avail_dpr].first().reset_index() if avail_dpr else None

    del df
    gc.collect()

    # ── Merge all aggregations ────────────────────────────────────────────────
    print("  Merging aggregations …")
    result = a1
    for other in [a2, a3, a4, a4_up, a4_dn, a5, a6, a7] + opt:
        result = result.merge(other, on=KEY, how="left")
    if dpr_agg is not None:
        result = result.merge(dpr_agg, on=KEY, how="left")
    del a1, a2, a3, a4, a4_up, a4_dn, a5, a6, a7, opt, dpr_agg
    gc.collect()

    # ── Derived time features ─────────────────────────────────────────────────
    H   = 3600.0
    CAP = SHIFT_CAP_HRS
    for raw_col, hr_col in [
        ("engine_on_s",  "engine_on_hrs"),
        ("engine_off_s", "engine_off_hrs"),
        ("prod_s",       "prod_hrs"),
        ("idle_s",       "idle_hrs"),
        ("maint_s",      "maint_hrs"),
        ("blind_s",      "blind_gap_hrs"),
    ]:
        result[hr_col] = (result[raw_col] / H).round(4).clip(upper=CAP)

    # prod + idle must not exceed engine_on
    tot_on = result["prod_hrs"] + result["idle_hrs"]
    over   = tot_on > result["engine_on_hrs"]
    if over.any():
        scale = result.loc[over, "engine_on_hrs"] / tot_on.loc[over]
        result.loc[over, "prod_hrs"] = (result.loc[over, "prod_hrs"] * scale).round(4)
        result.loc[over, "idle_hrs"] = (result.loc[over, "idle_hrs"] * scale).round(4)

    result["idle_ratio"] = (
        result["idle_hrs"] / result["engine_on_hrs"].replace(0, np.nan)
    ).fillna(0).round(4)

    # Distance / elevation rounding
    for col in ["total_dist_spd_km", "total_dist_hav_km", "dist_steep_spd_km",
                "dist_steep_hav_km", "uphill_dist_hav_km", "downhill_dist_hav_km"]:
        result[col] = result[col].fillna(0).round(4)
    for col in ["net_lift_m", "net_descent_m", "gross_elev_m"]:
        result[col] = result[col].round(2)
    result["alt_std"] = result["alt_std"].fillna(0).round(4)

    # Gradient / angle / speed rounding
    for col in ["mean_gradient_pct", "max_gradient_pct", "std_grad",
                "mean_uphill_grad", "mean_downhill_grad"]:
        result[col] = result[col].fillna(0).round(4)
    for col in ["mean_angle_change", "std_angle_change", "total_turning_deg"]:
        result[col] = result[col].fillna(0).round(4)
    result["n_sharp_turns"] = result["n_sharp_turns"].fillna(0).astype(int)
    for col in ["mean_speed", "max_speed", "std_speed"]:
        result[col] = result[col].fillna(0).round(4)

    result["gps_ok_pct"] = (result["gps_ok_pct"] * 100).round(2)

    # Stoppage density
    ref_dist = result["total_dist_hav_km"].replace(0, np.nan)
    result["n_stops"]           = result["n_stops"].clip(lower=0).astype(int)
    result["stopgo_intensity"]  = (result["n_stops"] / ref_dist).fillna(0).round(4)
    result["stoppage_per_hour"] = (result["n_stops"] / result["engine_on_hrs"].replace(0, np.nan)).fillna(0).round(4)
    result["dist_per_stop"]     = (ref_dist / result["n_stops"].replace(0, np.nan)).fillna(0).round(4)
    result["elev_per_dist"]     = (result["gross_elev_m"] / ref_dist).fillna(0).round(4)

    result.rename(columns={"vehicle": "dumper_id", "shift_dpr": "shift"}, inplace=True)

    # Fill missing operator_id with mode per vehicle
    prim = (
        result[result["operator_id"].notna()]
        .groupby("dumper_id")["operator_id"]
        .agg(lambda x: x.mode().iloc[0] if len(x) else np.nan)
    )
    result["operator_id"] = result.apply(
        lambda r: prim.get(r["dumper_id"], np.nan) if pd.isna(r["operator_id"]) else r["operator_id"],
        axis=1,
    )
    result["operator_id"] = result["operator_id"].fillna(-1).astype(int)

    # Drop raw second columns
    result.drop(
        columns=[c for c in result.columns if c in
                 ["engine_on_s", "engine_off_s", "prod_s", "idle_s", "maint_s", "blind_s"]],
        inplace=True,
    )

    # ── DPR sanity checks (train only) ────────────────────────────────────────
    if is_train and "prod_hr_dpr" in result.columns:
        _sanity_check(result)

    # Strip DPR columns before saving
    feat_cols = [c for c in result.columns if c not in DPR_COLS]
    result[feat_cols].to_parquet(output_path, index=False)
    print(f"\n  ✅  Saved → {output_path.split('/')[-1]}  {result[feat_cols].shape}")
    del result
    gc.collect()


def _sanity_check(result: pd.DataFrame) -> None:
    """Print correlation / MAE between calculated features and DPR ground truth."""

    def _chk(df, calc, ref, unit="", lbl=""):
        if ref not in df.columns:
            return
        v  = df[[calc, ref]].dropna()
        v  = v[(v[ref] > 0) | (v[calc] > 0)]
        if not len(v):
            return
        d  = v[calc] - v[ref]
        pe = (d / v[ref].replace(0, np.nan)).abs() * 100
        print(f"\n  {lbl}  [{calc}] vs [{ref}]  (n={len(v):,})")
        print(f"    Corr  : {v[calc].corr(v[ref]):.4f}")
        print(f"    MAE   : {d.abs().mean():.4f} {unit}")
        print(f"    Bias  : {d.median():.4f} {unit}  (+ve = calc > DPR)")
        print(f"    ≤10%  : {(pe<=10).mean()*100:.1f}%  |  ≤20%: {(pe<=20).mean()*100:.1f}%")

    print("\n" + "=" * 70)
    print("  SANITY CHECKS — Calculated vs DPR Ground Truth")
    print("=" * 70)
    _chk(result, "prod_hrs",          "prod_hr_dpr",  "hrs", "🕐")
    _chk(result, "idle_hrs",          "idle_hr_dpr",  "hrs", "🕐")
    _chk(result, "maint_hrs",         "maint_hr_dpr", "hrs", "🔧")
    _chk(result, "total_dist_hav_km", "km_dpr",       "km",  "📏 HAV dist")
    _chk(result, "total_dist_spd_km", "km_dpr",       "km",  "📏 SPD dist")
    if "cumdist_range" in result.columns:
        _chk(result, "total_dist_hav_km", "cumdist_range", "km", "📏 cumdist")


if __name__ == "__main__":
    # Block 3 — train
    build_shift_features(
        OUT_DIR + "train_dump_preproc.parquet",
        OUT_DIR + "train_features.parquet",
        is_train=True,
    )
    # Block 6 — test
    build_shift_features(
        OUT_DIR + "test_dump_preproc.parquet",
        OUT_DIR + "test_features.parquet",
        is_train=False,
    )

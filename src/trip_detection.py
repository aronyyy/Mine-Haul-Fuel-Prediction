"""
Blocks 4 & 7 — Trip detection.

Loading events are detected via 3-D spatio-temporal matching against
Exc*/WHL*/BHL* loader pings.  Dump events use three methods depending
on vehicle metadata:
  (a) analog_input_1 body-up transitions (analog-sensor vehicles after Feb-17)
  (b) 3-D GeoJSON zone match (mine001 geofence vehicles)
  (c) 50/50 haul-distance split fallback (mine002 / unknown)

Both loaded_dist_spd_km and loaded_dist_hav_km are produced.

Outputs (written to data/processed/):
  train_trip_features.parquet
  test_trip_features.parquet
"""

import gc
import os
import warnings

import numpy as np
import pandas as pd

from config import (
    ANALOG_CUT, DUMP_BODY_THR, DUMP_ZONES_FILE, OUT_DIR,
    load_fleet_meta, safe_div,
)

warnings.filterwarnings("ignore")

KEY_SM = ["vehicle", "date_dpr", "shift_dpr"]


def detect_trips(
    dump_preproc:   str,
    loader_preproc: str,
    feats_path:     str,
    output_path:    str,
) -> None:
    """
    Run the full trip-detection pipeline for one split (train or test).

    Parameters
    ----------
    dump_preproc   : Path to *_dump_preproc.parquet.
    loader_preproc : Path to *_loaders_preproc.parquet (may not exist for test).
    feats_path     : Path to *_features.parquet (needed for mine002 50/50 fallback).
    output_path    : Destination path for *_trip_features.parquet.
    """
    print(f"\n  Loading dump preproc …")
    dump_df = pd.read_parquet(dump_preproc)
    dump_df["ts"] = pd.to_datetime(dump_df["ts"], utc=True).astype("datetime64[ns, UTC]")
    dump_df = dump_df.sort_values(["vehicle", "ts"]).reset_index(drop=True)
    print(f"  Dump shape: {dump_df.shape}")

    # ── Re-derive distance ────────────────────────────────────────────────────
    gps = dump_df["gps_ok"].astype(bool).values
    spd = dump_df["speed"].values
    ign = dump_df["ignition"].values == 1
    dt  = dump_df["dt_sec"].values
    gap = dump_df["is_gap_300s"].values == 1.0
    mv  = ign & (spd >= 2) & gps & ~gap

    dump_df["_d_spd"] = np.where(mv, (spd * dt) / 3600, 0).astype(np.float32)
    if "disthav" in dump_df.columns:
        dh = dump_df["disthav"].fillna(0).values
        dump_df["_d_hav"] = np.where(gps & (dh > 0), dh / 1000, 0).astype(np.float32)
    else:
        dump_df["_d_hav"] = dump_df["_d_spd"].copy()

    adiff = dump_df.groupby("vehicle")["altitude"].diff().fillna(0).values.astype(np.float32)
    adiff[gap] = 0.0
    dump_df["_adiff"] = adiff

    # ── Fleet metadata ────────────────────────────────────────────────────────
    analog_vehs, geo_vehs, m2_vehs = load_fleet_meta()

    # ── Stop event extraction ─────────────────────────────────────────────────
    print("  Extracting stop events …")
    dump_df["is_stp"] = (dump_df["speed"] < 3.5) & (dump_df["ignition"] == 1)
    blk_change = dump_df["is_stp"] != dump_df.groupby("vehicle", sort=False)["is_stp"].shift(1)
    dump_df["stop_blk"] = blk_change.cumsum().astype(np.int32)

    stops = (
        dump_df[dump_df["is_stp"]]
        .groupby(["vehicle", "date_dpr", "shift_dpr", "stop_blk"], observed=True)
        .agg(
            start_ts  = ("ts",       "min"),
            end_ts    = ("ts",       "max"),
            d_utm_x   = ("utm_x",    "median"),
            d_utm_y   = ("utm_y",    "median"),
            d_alt     = ("altitude", "median"),
            mine_anon = ("mine_anon","first"),
        )
        .reset_index()
    )
    stops["dur_sec"] = (stops["end_ts"] - stops["start_ts"]).dt.total_seconds()
    stops["mid_ts"]  = (
        stops["start_ts"] + pd.to_timedelta(stops["dur_sec"] / 2, unit="s")
    ).astype("datetime64[ns, UTC]")
    gc.collect()

    # ── Loading detection (3-D spatio-temporal) ───────────────────────────────
    print("  Loading detection (3D spatio-temporal) …")
    load_cands = stops[(stops["dur_sec"] >= 45) & (stops["dur_sec"] <= 1200)].sort_values("mid_ts")
    conf_loads = pd.DataFrame()

    if os.path.exists(loader_preproc):
        ldr = pd.read_parquet(
            loader_preproc,
            columns=["vehicle", "ts", "utm_x", "utm_y", "altitude"],
        )
        ldr["ts"] = pd.to_datetime(ldr["ts"], utc=True).astype("datetime64[ns, UTC]")
        ldr = ldr.sort_values("ts").rename(
            columns={"utm_x": "l_x", "utm_y": "l_y", "altitude": "l_alt", "vehicle": "l_id"}
        )
        all_m = []
        for l_id, l_data in ldr.groupby("l_id"):
            m = pd.merge_asof(
                load_cands,
                l_data[["ts", "l_id", "l_x", "l_y", "l_alt"]],
                left_on="mid_ts", right_on="ts",
                direction="nearest", tolerance=pd.Timedelta("2min"),
            ).dropna(subset=["l_id"])
            m["d2d"] = np.sqrt((m["d_utm_x"] - m["l_x"]) ** 2 + (m["d_utm_y"] - m["l_y"]) ** 2)
            m["dz"]  = (m["d_alt"] - m["l_alt"]).abs()
            all_m.append(m)
        del ldr
        gc.collect()

        if all_m:
            all_m = pd.concat(all_m, ignore_index=True)
            all_m = all_m.sort_values(["vehicle", "stop_blk", "d2d"])
            conf_loads = (
                all_m.drop_duplicates(["vehicle", "stop_blk"], keep="first")
                .query("d2d <= 50 and dz <= 15")
                .copy()
            )
        del all_m
        gc.collect()
        print(f"    Confirmed loads: {len(conf_loads):,}")
    else:
        print("  ⚠️  No loader preproc found; loading match skipped.")

    load_end_ts = set(conf_loads["end_ts"].values) if len(conf_loads) else set()

    # ── Dump detection ────────────────────────────────────────────────────────
    print("  Dump detection …")
    dump_cands = stops[(stops["dur_sec"] >= 15) & (stops["dur_sec"] <= 600)]

    # (a) Analog body-up transitions
    print("    → Analog body-up transitions …")
    analog_dump_ts = set()
    if "analog_input_1" in dump_df.columns:
        for veh in analog_vehs:
            vdf = dump_df[(dump_df["vehicle"] == veh) & (dump_df["ts"] >= ANALOG_CUT)].sort_values("ts")
            if not len(vdf):
                continue
            ai     = vdf["analog_input_1"].fillna(0).values
            bu     = (ai > DUMP_BODY_THR).astype(int)
            ch     = np.diff(bu, prepend=bu[0])
            si_lst = np.where(ch == 1)[0]
            ei_lst = np.where(ch == -1)[0]
            ts_arr = vdf["ts"].values
            for si in si_lst:
                fut = ei_lst[ei_lst > si]
                if not len(fut):
                    continue
                ei  = fut[0]
                dur = (ts_arr[ei] - ts_arr[si]) / np.timedelta64(1, "s")
                if 10 <= dur <= 600:
                    analog_dump_ts.add(ts_arr[ei])
    print(f"    Analog dump events: {len(analog_dump_ts):,}")

    # (b) Geofence (mine001 non-analog vehicles)
    print("    → Geofence zones …")
    geo_dump_ts = set()
    if os.path.exists(DUMP_ZONES_FILE):
        try:
            import geopandas as gpd
            geo_c = dump_cands[
                dump_cands["vehicle"].isin(geo_vehs)
                | ((dump_cands["mine_anon"] == "mine001") & ~dump_cands["vehicle"].isin(analog_vehs))
            ].copy()
            if len(geo_c):
                gdf_s = gpd.GeoDataFrame(
                    geo_c,
                    geometry=gpd.points_from_xy(geo_c.d_utm_x, geo_c.d_utm_y),
                    crs="EPSG:32645",
                )
                gdf_z = gpd.read_file(DUMP_ZONES_FILE)
                jnd   = gpd.sjoin(gdf_s, gdf_z, how="inner", predicate="intersects")
                if "median_altitude" in gdf_z.columns:
                    jnd = jnd[(jnd["d_alt"] - jnd["median_altitude"]).abs() <= 20]
                geo_dump_ts.update(jnd["end_ts"].values)
            print(f"    Geofence dump events: {len(geo_dump_ts):,}")
        except Exception as ex:
            print(f"    ⚠️  Geofence error: {ex}")
    else:
        print("    ⚠️  Dump zones file not found.")

    all_dump_ts = analog_dump_ts | geo_dump_ts

    # ── State machine (loaded / empty) ────────────────────────────────────────
    print("  Building state machine …")
    ts_ns = dump_df["ts"].values.astype("datetime64[ns]").astype(np.int64)

    def _mark(ev_set, tol=5_000_000_000):
        if not ev_set:
            return np.full(len(dump_df), np.nan)
        ev  = np.sort(np.fromiter(
            (t.astype(np.int64) if hasattr(t, "astype") else int(t) for t in ev_set),
            dtype=np.int64,
        ))
        idx = np.searchsorted(ev, ts_ns, side="left").clip(0, len(ev) - 1)
        il  = (idx - 1).clip(0, len(ev) - 1)
        md  = np.minimum(np.abs(ts_ns - ev[idx]), np.abs(ts_ns - ev[il]))
        return np.where(md <= tol, 1.0, np.nan)

    lf      = _mark(load_end_ts)
    df_flag = _mark(all_dump_ts)

    dump_df["ev_flag"] = np.where(~np.isnan(lf), 1.0, np.where(~np.isnan(df_flag), 0.0, np.nan))
    dump_df["state"]   = (
        dump_df.groupby(KEY_SM, observed=True)["ev_flag"]
        .transform(lambda s: s.ffill())
        .fillna(0)
    )
    dump_df["is_load"] = dump_df["state"] == 1
    prev_st = dump_df.groupby(KEY_SM, observed=True)["state"].shift(1).fillna(0)
    dump_df["new_trip"] = ((dump_df["state"] == 1) & (prev_st == 0)).astype(int)

    # ── Loaded / empty distance & elevation ───────────────────────────────────
    il  = dump_df["is_load"].values
    adv = dump_df["_adiff"].values
    mv2 = (dump_df["ignition"].values == 1) & (spd >= 2)

    dump_df["_ld_spd"]   = np.where(il,   dump_df["_d_spd"].values, 0).astype(np.float32)
    dump_df["_em_spd"]   = np.where(~il,  dump_df["_d_spd"].values, 0).astype(np.float32)
    dump_df["_ld_hav"]   = np.where(il,   dump_df["_d_hav"].values, 0).astype(np.float32)
    dump_df["_em_hav"]   = np.where(~il,  dump_df["_d_hav"].values, 0).astype(np.float32)
    dump_df["_ld_climb"] = np.where(il  & (adv > 0),  adv, 0).astype(np.float32)
    dump_df["_em_climb"] = np.where(~il & (adv > 0),  adv, 0).astype(np.float32)
    dump_df["_ld_desc"]  = np.where(il  & (adv < 0), -adv, 0).astype(np.float32)
    dump_df["_ld_spd_v"] = np.where(il  & mv2, dump_df["speed"].values, np.nan).astype(np.float32)
    dump_df["_em_spd_v"] = np.where(~il & mv2, dump_df["speed"].values, np.nan).astype(np.float32)

    # ── Aggregation ───────────────────────────────────────────────────────────
    print("  Aggregating trip features …")
    sf = dump_df.groupby(KEY_SM, observed=True).agg(
        calc_trips     = ("new_trip",   "sum"),
        _ld_spd        = ("_ld_spd",   "sum"),
        _em_spd        = ("_em_spd",   "sum"),
        _ld_hav        = ("_ld_hav",   "sum"),
        _em_hav        = ("_em_hav",   "sum"),
        loaded_climb_m = ("_ld_climb", "sum"),
        empty_climb_m  = ("_em_climb", "sum"),
        loaded_desc_m  = ("_ld_desc",  "sum"),
        _ld_sv         = ("_ld_spd_v", "sum"),
        _ld_sc         = ("_ld_spd_v", "count"),
        _em_sv         = ("_em_spd_v", "sum"),
        _em_sc         = ("_em_spd_v", "count"),
        mine_anon      = ("mine_anon", "first"),
    ).reset_index()

    # Loading event metadata
    if len(conf_loads) and "shift_dpr" in conf_loads.columns:
        la = (
            conf_loads
            .groupby(["vehicle", "date_dpr", "shift_dpr"], observed=True)
            .agg(
                avg_loading_time_sec = ("dur_sec", "mean"),
                n_unique_loaders     = ("l_id",    "nunique"),
            )
            .reset_index()
        )
        sf = sf.merge(la, on=KEY_SM, how="left")

    # Cycle time
    tr = dump_df[dump_df["new_trip"] == 1].copy()
    if len(tr):
        tr["ct"] = tr.groupby(KEY_SM, observed=True)["ts"].diff().dt.total_seconds()
        cyc = tr.groupby(KEY_SM, observed=True).agg(
            avg_cycle_sec = ("ct", "mean"),
            std_cycle_sec = ("ct", "std"),
        ).reset_index()
        sf = sf.merge(cyc, on=KEY_SM, how="left")

    del dump_df, tr
    gc.collect()

    # ── Derived ratio / speed features ────────────────────────────────────────
    sf["loaded_dist_spd_km"] = sf["_ld_spd"].round(4)
    sf["empty_dist_spd_km"]  = sf["_em_spd"].round(4)
    sf["loaded_dist_hav_km"] = sf["_ld_hav"].round(4)
    sf["empty_dist_hav_km"]  = sf["_em_hav"].round(4)
    tot_hav = sf["_ld_hav"] + sf["_em_hav"]
    tot_spd = sf["_ld_spd"] + sf["_em_spd"]
    sf["loaded_ratio_hav"]  = safe_div(sf["_ld_hav"], tot_hav).round(4)
    sf["loaded_ratio_spd"]  = safe_div(sf["_ld_spd"], tot_spd).round(4)
    sf["avg_speed_loaded"]  = safe_div(sf["_ld_sv"], sf["_ld_sc"]).round(4)
    sf["avg_speed_empty"]   = safe_div(sf["_em_sv"], sf["_em_sc"]).round(4)
    sf["speed_ratio_l_e"]   = safe_div(sf["avg_speed_loaded"], sf["avg_speed_empty"]).round(4)
    if "avg_cycle_sec" in sf.columns:
        sf["trips_per_hour"] = safe_div(
            sf["calc_trips"],
            sf["avg_cycle_sec"] * sf["calc_trips"] / 3600,
        ).round(4)

    # ── Mine002 / unknown → 50/50 fallback ───────────────────────────────────
    all_known = analog_vehs | geo_vehs | m2_vehs
    m2_mask   = sf["mine_anon"].isin(m2_vehs) | ~sf["vehicle"].isin(all_known)

    feats = pd.read_parquet(
        feats_path,
        columns=["dumper_id", "date_dpr", "shift", "total_dist_hav_km", "total_dist_spd_km"],
    ).rename(columns={"dumper_id": "vehicle", "shift": "shift_dpr"})
    sf = sf.merge(feats, on=["vehicle", "date_dpr", "shift_dpr"], how="left")
    del feats
    gc.collect()

    for col, src in [
        ("loaded_dist_hav_km", "total_dist_hav_km"),
        ("empty_dist_hav_km",  "total_dist_hav_km"),
        ("loaded_dist_spd_km", "total_dist_spd_km"),
        ("empty_dist_spd_km",  "total_dist_spd_km"),
    ]:
        sf.loc[m2_mask, col] = sf.loc[m2_mask, src] * 0.5
    for col in ["loaded_ratio_hav", "loaded_ratio_spd"]:
        sf.loc[m2_mask, col] = 0.5
    for col in ["loaded_climb_m", "empty_climb_m", "loaded_desc_m",
                "avg_loading_time_sec", "n_unique_loaders"]:
        if col in sf.columns:
            sf.loc[m2_mask, col] = np.nan

    sf.drop(
        columns=[c for c in sf.columns if c in [
            "_ld_spd", "_em_spd", "_ld_hav", "_em_hav",
            "_ld_sv", "_ld_sc", "_em_sv", "_em_sc",
            "total_dist_hav_km", "total_dist_spd_km", "mine_anon",
        ]],
        inplace=True, errors="ignore",
    )

    sf.rename(columns={"vehicle": "dumper_id", "shift_dpr": "shift"}, inplace=True)
    sf.to_parquet(output_path, index=False)
    print(f"  ✅  Saved → {output_path.split('/')[-1]}  {sf.shape}")
    del sf
    gc.collect()


if __name__ == "__main__":
    # Block 4 — train
    detect_trips(
        OUT_DIR + "train_dump_preproc.parquet",
        OUT_DIR + "train_loaders_preproc.parquet",
        OUT_DIR + "train_features.parquet",
        OUT_DIR + "train_trip_features.parquet",
    )
    # Block 7 — test
    detect_trips(
        OUT_DIR + "test_dump_preproc.parquet",
        OUT_DIR + "test_loaders_preproc.parquet",
        OUT_DIR + "test_features.parquet",
        OUT_DIR + "test_trip_features.parquet",
    )

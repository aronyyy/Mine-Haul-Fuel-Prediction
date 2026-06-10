"""
Blocks 1 & 2 — Raw telemetry pre-processing.

Reads raw parquet telemetry files, separates Dump* from Loader vehicles,
applies GPS quality flags, clips speed/altitude, projects to UTM, computes
time deltas, and classifies data gaps.

Outputs (written to data/processed/):
  train_dump_preproc.parquet
  train_loaders_preproc.parquet
  test_dump_preproc.parquet
  test_loaders_preproc.parquet
"""

import gc
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer

from config import (
    ALT_MAX, ALT_MIN, DUMP_PREFIX, GPS_HDOP_LIM, GPS_PDOP_LIM,
    GPS_SAT_MIN, LOADER_PREFIXES, OUT_DIR, SPEED_MAX,
    TEST_FILES, TRAIN_FILES, reduce_mem,
)

warnings.filterwarnings("ignore")


# ── Shared CRS transformer (EPSG:4326 → UTM Zone 45N) ─────────────────────────
_TF = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)


def _preprocess_dump(d: pd.DataFrame) -> pd.DataFrame:
    """Apply GPS flags, speed/altitude clipping, and UTM projection to Dump* rows."""
    eff = d["gnss_hdop"].fillna(d["gnss_pdop"])
    d["gps_ok"] = (
        (d["satellites"] >= GPS_SAT_MIN)
        & (eff <= GPS_HDOP_LIM)
        & (d["gnss_pdop"] <= GPS_PDOP_LIM)
    ).astype(np.int8)
    d["speed"]    = d["speed"].clip(0, SPEED_MAX).astype(np.float32)
    d["altitude"] = d["altitude"].clip(ALT_MIN, ALT_MAX).astype(np.float32)
    ux, uy        = _TF.transform(d["longitude"].values, d["latitude"].values)
    d["utm_x"]    = ux.astype(np.float32)
    d["utm_y"]    = uy.astype(np.float32)
    for c in ["gsm_operator", "mine_anon", "shift_dpr", "date_dpr"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
    return reduce_mem(d)


def _preprocess_loader(e: pd.DataFrame) -> pd.DataFrame:
    """Apply UTM projection to Loader rows."""
    ux, uy     = _TF.transform(e["longitude"].values, e["latitude"].values)
    e["utm_x"] = ux.astype(np.float32)
    e["utm_y"] = uy.astype(np.float32)
    for c in ["gsm_operator", "mine_anon", "shift_dpr", "date_dpr"]:
        if c in e.columns:
            e[c] = e[c].astype(str).str.strip()
    return reduce_mem(e)


def _add_time_deltas(df: pd.DataFrame, fill_gaps_with: float = 0.0) -> pd.DataFrame:
    """Compute actual_dt_sec, is_gap_300s, and dt_sec (capped at 300 s)."""
    df["actual_dt_sec"] = (
        df.groupby("vehicle")["ts"]
        .diff()
        .dt.total_seconds()
        .fillna(fill_gaps_with)
        .astype(np.float32)
    )
    df["is_gap_300s"] = (df["actual_dt_sec"] >= 300).astype(np.float32)
    df["dt_sec"]      = df["actual_dt_sec"].clip(upper=300).astype(np.float32)
    return df


def _classify_gap_status(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each data gap as normal / engine_off_stationary / engine_running_blind."""
    gap_rows = df["is_gap_300s"] == 1.0
    if "fuel_volume" in df.columns:
        pf = df.groupby("vehicle")["fuel_volume"].shift(1)
        df["fuel_delta"] = np.where(
            gap_rows, df["fuel_volume"] - pf, np.nan
        ).astype(np.float32)
        del pf
        df["gap_status"] = "normal"
        df.loc[gap_rows & (df["fuel_delta"] >= -2.0), "gap_status"] = "engine_off_stationary"
        df.loc[gap_rows & (df["fuel_delta"] <  -2.0), "gap_status"] = "engine_running_blind"
    else:
        df["gap_status"] = "normal"
    return df


def run_preprocess(file_list: list, tag: str, is_train: bool) -> None:
    """
    Process a list of telemetry parquets and write preprocessed outputs.

    Parameters
    ----------
    file_list : list of str
        Paths to raw parquet files.
    tag : str
        'train' or 'test' — used as output filename prefix.
    is_train : bool
        If True, fuel_delta / gap_status are computed from fuel_volume column.
    """
    print("=" * 70)
    print(f"{'TRAIN' if is_train else 'TEST'} RAW PRE-PROCESSING")
    print("=" * 70)

    dump_chunks, ldr_chunks = [], []

    for fpath in file_list:
        print(f"\n  ↪  {fpath.split('/')[-1]}")
        raw = pd.read_parquet(fpath)
        vs  = raw["vehicle"].astype(str)
        d   = raw[vs.str.startswith(DUMP_PREFIX)].copy()
        e   = raw[vs.str.startswith(LOADER_PREFIXES)].copy()
        del raw, vs
        gc.collect()
        print(f"       Dump*: {len(d):,}   Loaders: {len(e):,}")

        if len(d):
            dump_chunks.append(_preprocess_dump(d))
            del d

        if len(e):
            ldr_chunks.append(_preprocess_loader(e))
            del e

        gc.collect()

    # ── Dump* ─────────────────────────────────────────────────────────────────
    print("\n  Combining Dump* …")
    df = pd.concat(dump_chunks, ignore_index=True)
    del dump_chunks
    gc.collect()
    df = df.sort_values(["vehicle", "ts"]).reset_index(drop=True)

    print("  Computing time deltas …")
    fill = 0.0 if is_train else 5.0
    df = _add_time_deltas(df, fill_gaps_with=fill)

    if is_train:
        print("  Computing gap_status …")
        df = _classify_gap_status(df)

    out_dump = OUT_DIR + f"{tag}_dump_preproc.parquet"
    print(f"\n  Dump* shape : {df.shape}")
    print(f"  Memory      : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  GPS OK      : {df['gps_ok'].mean() * 100:.1f}%")
    df.to_parquet(out_dump, index=False)
    print(f"  ✅  {out_dump}")
    del df
    gc.collect()

    # ── Loaders ───────────────────────────────────────────────────────────────
    if ldr_chunks:
        ldr = pd.concat(ldr_chunks, ignore_index=True)
        del ldr_chunks
        gc.collect()
        ldr = ldr.sort_values(["vehicle", "ts"]).reset_index(drop=True)
        out_ldr = OUT_DIR + f"{tag}_loaders_preproc.parquet"
        ldr.to_parquet(out_ldr, index=False)
        print(f"  ✅  {out_ldr}  ({len(ldr):,} rows)")
        del ldr
        gc.collect()

    print(f"\n  {'TRAIN' if is_train else 'TEST'} PRE-PROCESSING COMPLETE ✅")


if __name__ == "__main__":
    run_preprocess(TRAIN_FILES, tag="train", is_train=True)
    run_preprocess(TEST_FILES,  tag="test",  is_train=False)

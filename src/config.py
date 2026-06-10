"""
Global constants, file paths, and shared utility functions.
All raw inputs are read from data/raw/, all outputs written to data/processed/.
"""

import pandas as pd
import numpy as np
import gc

# ── Directories ───────────────────────────────────────────────────────────────
RAW_DIR  = "data/raw/"
OUT_DIR  = "data/processed/"

DUMP_ZONES_FILE = RAW_DIR + "mine001_dump_zones_3d.geojson"

# ── Physics / GPS thresholds ──────────────────────────────────────────────────
SHIFT_CAP_HRS = 8.0
GPS_PDOP_LIM  = 6.0
GPS_HDOP_LIM  = 4.0
GPS_SAT_MIN   = 5
SPEED_MAX     = 60.0
ALT_MIN       = 500.0
ALT_MAX       = 800.0
ANALOG_CUT    = pd.Timestamp("2026-02-17", tz="UTC")
DUMP_BODY_THR = 0.5   # analog_input_1 > this → dump body raised

# ── Input file lists ──────────────────────────────────────────────────────────
TRAIN_FILES = [
    RAW_DIR + "telemetry_2026-01-01_2026-01-10.parquet",
    RAW_DIR + "telemetry_2026-01-11_2026-01-20.parquet",
    RAW_DIR + "telemetry_2026-02-01_2026-02-10.parquet",
    RAW_DIR + "telemetry_2026-02-11_2026-02-20.parquet",
    RAW_DIR + "telemetry_2026-03-01_2026-03-11.parquet",
]
TEST_FILES = [
    RAW_DIR + "telemetry_2026-01-21_2026-01-31.parquet",
    RAW_DIR + "telemetry_2026-02-21_2026-02-28.parquet",
    RAW_DIR + "telemetry_2026-03-12_2026-03-20.parquet",
]
FUEL_FILES = [
    RAW_DIR + "smry_jan_train_ordered.csv",
    RAW_DIR + "smry_feb_train_ordered.csv",
    RAW_DIR + "smry_mar_train_ordered.csv",
]

# ── Column groupings ──────────────────────────────────────────────────────────
DPR_COLS        = ["prod_hr_dpr", "idle_hr_dpr", "maint_hr_dpr",
                   "km_dpr", "tonnage", "bd_hr_dpr", "hmr_dpr"]
LOADER_PREFIXES = ("Exc", "WHL", "BHL")
DUMP_PREFIX     = "Dump"

# ── LSTM hyper-parameters ─────────────────────────────────────────────────────
HIDDEN_DIM  = 16
MAX_SEQ_LEN = 600
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-3
DROPOUT     = 0.2

SEQ_FEATURES = [
    "speed",
    "altitude",
    "ignition",
    "gps_ok",
    "log_dt",
    "is_moving",
    "adiff_norm",
    "spd_x_dt",
]
N_FEATS = len(SEQ_FEATURES)


# ── Shared helper functions ───────────────────────────────────────────────────

def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes; convert low-cardinality objects to category."""
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            continue
        if dt == object:
            if df[col].nunique() / max(len(df), 1) < 0.5:
                df[col] = df[col].astype("category")
            continue
        if pd.api.types.is_numeric_dtype(dt):
            mn, mx = df[col].min(), df[col].max()
            if pd.isna(mn) or pd.isna(mx):
                df[col] = df[col].astype(np.float32)
                continue
            if str(dt).startswith("int"):
                for t in [np.int8, np.int16, np.int32]:
                    if mn > np.iinfo(t).min and mx < np.iinfo(t).max:
                        df[col] = df[col].astype(t)
                        break
            else:
                df[col] = df[col].astype(
                    np.float32
                    if mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max
                    else np.float64
                )
    return df


def safe_div(a, b, fill: float = 0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, fill)


def load_fleet_meta():
    """Return three sets: analog-sensor vehicles, geofence vehicles, mine002 vehicles."""
    fl = pd.read_csv(RAW_DIR + "fleet.csv")
    fl.columns = fl.columns.str.strip().str.lower()
    analog = set(fl.loc[fl["dump_switch"] == 1, "vehicle"])
    geo = set(fl.loc[
        fl["fleet"].str.lower().str.contains("dump")
        & (fl["mine_anon"] == "mine001")
        & fl["dump_switch"].isna(),
        "vehicle",
    ])
    m2 = set(fl.loc[fl["mine_anon"] == "mine002", "vehicle"])
    return analog, geo, m2


def load_fuel_labels() -> pd.DataFrame:
    """Load and concatenate all fuel summary CSVs; compute fuel_consumed_L."""
    parts = []
    for fp in FUEL_FILES:
        try:
            parts.append(pd.read_csv(fp))
        except FileNotFoundError:
            print(f"  ⚠️  {fp} not found")
    fuel = pd.concat(parts, ignore_index=True)
    fuel.columns = fuel.columns.str.strip().str.lower()
    fuel["fuel_consumed_L"] = (
        fuel["initlev"] + fuel["arefill"] - fuel["endlev"]
    ).clip(lower=0)
    fuel = fuel.rename(columns={"vehicle": "dumper_id", "date": "date_dpr", "shift": "shift"})
    fuel["date_dpr"] = pd.to_datetime(fuel["date_dpr"])
    return fuel[["dumper_id", "date_dpr", "shift", "fuel_consumed_L"]]

"""
TabPFN train + test — fuel consumption regression.

Uses a LightGBM scout to select the top-20 physical features,
then forces all LSTM embedding columns to be included.
Runs 5-fold TabPFN inference and exports a submission CSV.

Also produces secondary outputs:
  - route_benchmark (TabPFN without dumper/operator IDs)
  - efficiency_delta per shift
  - daily_consistency actuals vs predicted
  - secondary_output_charts.png (3-panel figure)

Inputs  (from data/processed/): train_final.parquet, test_final.parquet
Inputs  (from data/raw/):       id_mapping_new.csv, smry_*.csv
Outputs (to   data/processed/): final_submission_forced_lstm_pfn.csv
                                 submission_route_benchmark.csv
                                 submission_full_model.csv
                                 efficiency_delta.csv
                                 daily_consistency.csv
                                 secondary_output_charts.png
"""

import warnings
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
import tabpfn_client
from tabpfn_client import TabPFNRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from config import FUEL_FILES, OUT_DIR, RAW_DIR

warnings.filterwarnings("ignore")

TARGET    = "fuel_consumed_L"
JOIN_KEYS = ["dumper_id", "date_dpr", "shift"]

# Categorical columns (label-encoded before TabPFN)
CAT_FEATURES_RAW = ["dumper_id", "shift", "mine_id", "operator_id"]


# ── Feature preparation helper ────────────────────────────────────────────────

def prep_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exclude_ids: bool = False,
) -> tuple:
    """
    Select features, impute, scale, encode categories.

    Returns
    -------
    X_train_np, X_test_np, y_train, feature_names
    """
    drop_base = {TARGET, "had_refill", "runhrs", "lph",
                 "initlev", "endlev", "date_dpr", "id"}

    all_feats = [c for c in train.columns if c in test.columns and c not in drop_base]

    id_cols = ["dumper_id", "operator_id"]
    if exclude_ids:
        all_feats = [c for c in all_feats if c not in id_cols]

    cat_feats = [c for c in CAT_FEATURES_RAW if c in all_feats]
    num_feats = [c for c in all_feats if c not in cat_feats]

    # LightGBM scout for feature importance
    X_scout = train[all_feats].copy()
    for col in num_feats:
        med = X_scout[col].median()
        X_scout[col] = X_scout[col].fillna(med if not pd.isna(med) else 0).astype(np.float32)
    for c in cat_feats:
        X_scout[c] = X_scout[c].astype(str).fillna("Missing").astype("category")

    scout = lgb.LGBMRegressor(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1)
    scout.fit(X_scout, train[TARGET])

    imp_df     = pd.DataFrame({"f": all_feats, "imp": scout.feature_importances_})
    lstm_feats = [c for c in all_feats if "lstm" in c.lower()]
    non_lstm   = imp_df[~imp_df["f"].isin(lstm_feats)]
    top20_phys = non_lstm.sort_values("imp", ascending=False).head(20)["f"].tolist()

    # Force LSTM features in
    features  = top20_phys + lstm_feats
    for c in cat_feats:
        if c not in features:
            features.append(c)

    cat_feats = [c for c in CAT_FEATURES_RAW if c in features]
    num_feats = [c for c in features if c not in cat_feats]

    X_tr = train[features].copy()
    X_te = test[features].copy()

    for col in num_feats:
        med = train[col].median()
        X_tr[col] = X_tr[col].fillna(med if not pd.isna(med) else 0).astype(np.float32)
        X_te[col] = X_te[col].fillna(med if not pd.isna(med) else 0).astype(np.float32)

    scaler = StandardScaler()
    X_tr[num_feats] = scaler.fit_transform(X_tr[num_feats])
    X_te[num_feats] = scaler.transform(X_te[num_feats])

    for c in cat_feats:
        X_tr[c] = train[c].astype(str).fillna("Missing")
        X_te[c] = test[c].astype(str).fillna("Missing")
        le = LabelEncoder()
        le.fit(list(X_tr[c]) + list(X_te[c]))
        X_tr[c] = le.transform(X_tr[c])
        X_te[c] = le.transform(X_te[c])

    return X_tr.values, X_te.values, train[TARGET].values, features


def run_tabpfn_cv(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    tag: str = "",
) -> tuple:
    """5-fold TabPFN cross-validation.  Returns (oof_preds, test_preds, oof_rmse)."""
    kf          = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds   = np.zeros(len(y_tr))
    test_preds  = np.zeros(len(X_te))
    fold_scores = []

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_tr, y_tr)):
        model = TabPFNRegressor()
        model.fit(X_tr[t_idx], y_tr[t_idx])
        oof_preds[v_idx] = model.predict(X_tr[v_idx])
        test_preds      += model.predict(X_te) / kf.n_splits
        score = np.sqrt(mean_squared_error(y_tr[v_idx], oof_preds[v_idx]))
        fold_scores.append(score)
        print(f"  {tag} Fold {fold+1} RMSE: {score:.2f}")

    oof_rmse = np.sqrt(mean_squared_error(y_tr, oof_preds))
    print(f"  {tag} OOF RMSE: {oof_rmse:.2f}  (std: {np.std(fold_scores):.2f})")
    return np.clip(oof_preds, 0, None), np.clip(test_preds, 0, None), oof_rmse


# ── Main submission ───────────────────────────────────────────────────────────

def run_tabpfn_submission(api_token: str) -> None:
    """
    Full TabPFN pipeline with forced LSTM inclusion.

    Parameters
    ----------
    api_token : str  Your TabPFN API token from priorlabs.
    """
    print("=" * 65)
    print("TABPFN PIPELINE — Forced LSTM Inclusion")
    print("=" * 65)

    tabpfn_client.set_access_token(api_token)

    # Load data
    print("\n[1/4] Loading data …")
    train = pd.read_parquet(OUT_DIR + "train_final.parquet")
    test  = pd.read_parquet(OUT_DIR + "test_final.parquet")

    for df in [train, test]:
        for col in JOIN_KEYS:
            if col in df.columns:
                df[col] = df[col].astype(str)

    train = train[train["dumper_id"].str.startswith("Dump", na=False)]
    test  = test[test["dumper_id"].str.startswith("Dump",  na=False)]
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test  = test.reset_index(drop=True)
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

    # Feature selection
    print("\n[2/4] Feature selection (forced LSTM inclusion) …")
    X_tr, X_te, y_tr, features = prep_features(train, test, exclude_ids=False)
    lstm_feats = [c for c in features if "lstm" in c.lower()]
    print(f"  Total features: {len(features)}  (LSTM: {len(lstm_feats)})")

    # 5-fold TabPFN
    print("\n[3/4] Running 5-fold TabPFN …")
    oof_preds, test_preds, oof_rmse = run_tabpfn_cv(X_tr, y_tr, X_te, tag="[FULL]")
    print(f"\n  Overall OOF RMSE: {oof_rmse:.4f}")

    # Export submission
    print("\n[4/4] Formatting submission …")
    test["Predicted"] = np.clip(test_preds, a_min=0, a_max=None)

    mapping = pd.read_csv(RAW_DIR + "id_mapping_new.csv")
    mapping = mapping.rename(columns={"vehicle": "dumper_id", "date": "date_dpr"})
    for col in JOIN_KEYS:
        mapping[col] = mapping[col].astype(str)

    test_keys = test[JOIN_KEYS].copy()
    test_keys["Predicted"] = test["Predicted"]

    submission = pd.merge(mapping, test_keys, on=JOIN_KEYS, how="left")
    submission["Predicted"] = submission["Predicted"].fillna(train[TARGET].mean())
    out_path = OUT_DIR + "final_submission_forced_lstm_pfn.csv"
    submission[["id", "Predicted"]].sort_values("id").to_csv(out_path, index=False)
    print(f"  ✅  {out_path}")


# ── Secondary outputs ─────────────────────────────────────────────────────────

def run_secondary_outputs(api_token: str) -> None:
    """
    Produce route benchmark, efficiency delta, daily consistency,
    and a 3-panel chart for the report.

    Parameters
    ----------
    api_token : str  Your TabPFN API token from priorlabs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    print("=" * 65)
    print("SECONDARY OUTPUTS")
    print("=" * 65)

    tabpfn_client.set_access_token(api_token)

    train = pd.read_parquet(OUT_DIR + "train_final.parquet")
    test  = pd.read_parquet(OUT_DIR + "test_final.parquet")

    for df in [train, test]:
        for col in JOIN_KEYS:
            if col in df.columns:
                df[col] = df[col].astype(str)

    train = train[train["dumper_id"].str.startswith("Dump", na=False)]
    test  = test[test["dumper_id"].str.startswith("Dump",  na=False)]
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test  = test.reset_index(drop=True)
    print(f"\n  Train shifts: {len(train):,}  |  Test shifts: {len(test):,}")

    # 5.1 Route benchmark (no ID features)
    print("\n── 5.1  Route-Level Fuel Benchmark (no ID features) ──")
    X_tr_r, X_te_r, y_tr, _ = prep_features(train, test, exclude_ids=True)
    oof_route, test_route, rmse_route = run_tabpfn_cv(X_tr_r, y_tr, X_te_r, tag="[ROUTE]")
    train["pred_route"] = oof_route
    test["pred_route"]  = test_route

    # 5.2 Full model (with ID features)
    print("\n── 5.2  Dumper Efficiency Component (with ID features) ──")
    X_tr_f, X_te_f, y_tr, _ = prep_features(train, test, exclude_ids=False)
    oof_full, test_full, rmse_full = run_tabpfn_cv(X_tr_f, y_tr, X_te_f, tag="[FULL]")
    train["pred_full"]  = oof_full
    test["pred_full"]   = test_full

    train["efficiency_delta"] = train["pred_full"] - train["pred_route"]
    test["efficiency_delta"]  = test["pred_full"]  - test["pred_route"]

    print(f"\n  Route RMSE : {rmse_route:.2f} L")
    print(f"  Full RMSE  : {rmse_full:.2f} L")
    print(f"  Delta      : {rmse_route - rmse_full:.2f} L  "
          f"({100*(rmse_route-rmse_full)/rmse_route:.1f}% improvement)")

    eff_out = train[JOIN_KEYS + [TARGET, "pred_route", "pred_full", "efficiency_delta"]].copy()
    eff_out.to_csv(OUT_DIR + "efficiency_delta.csv", index=False)
    print("  Saved → efficiency_delta.csv")

    # 5.4 Daily consistency
    print("\n── 5.4  Daily Fuel Consistency ──")
    parts = []
    for fp in FUEL_FILES:
        try:
            p = pd.read_csv(fp)
            p.columns = p.columns.str.strip().str.lower()
            p["fuel_consumed_L"] = (p["initlev"] + p["arefill"] - p["endlev"]).clip(lower=0)
            parts.append(p[["vehicle", "date", "fuel_consumed_L"]])
        except Exception:
            pass
    actual_fuel = pd.concat(parts, ignore_index=True)
    actual_fuel["date"] = pd.to_datetime(actual_fuel["date"]).dt.date.astype(str)
    actual_daily = (
        actual_fuel.groupby(["vehicle", "date"])["fuel_consumed_L"].sum().reset_index()
        .rename(columns={"vehicle": "dumper_id", "date": "date_dpr", "fuel_consumed_L": "actual_daily_L"})
    )
    train["date_dpr_dt"] = pd.to_datetime(train["date_dpr"], errors="coerce").dt.date.astype(str)
    pred_daily = (
        train.groupby(["dumper_id", "date_dpr_dt"])["pred_full"].sum().reset_index()
        .rename(columns={"date_dpr_dt": "date_dpr", "pred_full": "pred_daily_L"})
    )
    daily = actual_daily.merge(pred_daily, on=["dumper_id", "date_dpr"], how="inner")
    daily["abs_error"] = (daily["pred_daily_L"] - daily["actual_daily_L"]).abs()
    daily["pct_error"] = daily["abs_error"] / (daily["actual_daily_L"] + 1e-6) * 100
    corr = daily["pred_daily_L"].corr(daily["actual_daily_L"])
    print(f"  Daily Pearson corr : {corr:.4f}")
    print(f"  Mean daily MAE     : {daily['abs_error'].mean():.1f} L")
    print(f"  Mean daily MAPE    : {daily['pct_error'].mean():.1f}%")
    daily.to_csv(OUT_DIR + "daily_consistency.csv", index=False)
    print("  Saved → daily_consistency.csv")

    # Charts
    print("\n── Generating charts ──")
    delta_by_dump = train.groupby("dumper_id")["efficiency_delta"].mean().sort_values(ascending=False)

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: scatter
    ax1    = fig.add_subplot(gs[0])
    sample = train.sample(min(500, len(train)), random_state=42)
    ax1.scatter(sample[TARGET], sample["pred_route"], alpha=0.45, s=20,
                color="#4C72B0", label=f"Route  (RMSE {rmse_route:.0f})")
    ax1.scatter(sample[TARGET], sample["pred_full"],  alpha=0.45, s=20,
                color="#DD8452", label=f"Full   (RMSE {rmse_full:.0f})")
    lims = [min(sample[TARGET].min(), sample["pred_route"].min(), sample["pred_full"].min()) * 0.95,
            max(sample[TARGET].max(), sample["pred_route"].max(), sample["pred_full"].max()) * 1.05]
    ax1.plot(lims, lims, "k--", lw=1, alpha=0.6, label="Perfect")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("Actual Fuel (L)"); ax1.set_ylabel("Predicted (L)")
    ax1.set_title("Route Benchmark vs Full Model\n(500 random training shifts)")
    ax1.legend(fontsize=8)

    # Panel 2: efficiency bar
    ax2    = fig.add_subplot(gs[1])
    n_show = min(15, len(delta_by_dump))
    show   = pd.concat([delta_by_dump.head(n_show), delta_by_dump.tail(n_show)]).drop_duplicates().sort_values(ascending=False)
    colours = ["#e74c3c" if v > 0 else "#2ecc71" for v in show.values]
    ax2.barh(show.index, show.values, color=colours, edgecolor="none")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Mean Efficiency Delta (L per shift)\n+ve = consumes more than route avg")
    ax2.set_title("Dumper Efficiency Component")
    ax2.tick_params(axis="y", labelsize=7)

    # Panel 3: daily consistency
    ax3       = fig.add_subplot(gs[2])
    daily_agg = (daily.groupby("date_dpr").agg(actual=("actual_daily_L", "sum"),
                                                predicted=("pred_daily_L", "sum"))
                 .reset_index().sort_values("date_dpr"))
    if len(daily_agg) > 30:
        daily_agg = daily_agg.iloc[:: max(1, len(daily_agg) // 30)].reset_index(drop=True)
    x = range(len(daily_agg))
    ax3.plot(x, daily_agg["actual"],    "o-", lw=1.5, ms=4, color="#2c7bb6", label="Actual")
    ax3.plot(x, daily_agg["predicted"], "s--", lw=1.5, ms=4, color="#d7191c", label="Predicted")
    ax3.fill_between(x, daily_agg["actual"], daily_agg["predicted"], alpha=0.12, color="grey")
    tick_step = max(1, len(daily_agg) // 8)
    ax3.set_xticks(list(x)[::tick_step])
    ax3.set_xticklabels(daily_agg["date_dpr"].iloc[::tick_step], rotation=40, ha="right", fontsize=7)
    ax3.set_xlabel("Date"); ax3.set_ylabel("Total Fleet Fuel (L)")
    ax3.set_title(f"Daily Fuel Consistency\nCorr={corr:.3f}  |  MAPE={daily['pct_error'].mean():.1f}%")
    ax3.legend(fontsize=8)

    plt.suptitle("KaRoNNN — Secondary Output Analysis", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart_path = OUT_DIR + "secondary_output_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → secondary_output_charts.png")

    # Submission CSVs
    mapping = pd.read_csv(RAW_DIR + "id_mapping_new.csv").rename(
        columns={"vehicle": "dumper_id", "date": "date_dpr"}
    )
    for col in JOIN_KEYS:
        mapping[col] = mapping[col].astype(str)

    test_keys = test[JOIN_KEYS].copy()
    test_keys["Predicted_route"] = test["pred_route"]
    test_keys["Predicted_full"]  = test["pred_full"]
    sub = mapping.merge(test_keys, on=JOIN_KEYS, how="left")
    sub = sub.fillna({"Predicted_route": train[TARGET].mean(), "Predicted_full": train[TARGET].mean()})

    sub[["id", "Predicted_route"]].rename(columns={"Predicted_route": "Predicted"}).sort_values("id").to_csv(
        OUT_DIR + "submission_route_benchmark.csv", index=False
    )
    sub[["id", "Predicted_full"]].rename(columns={"Predicted_full": "Predicted"}).sort_values("id").to_csv(
        OUT_DIR + "submission_full_model.csv", index=False
    )
    print("  Saved → submission_route_benchmark.csv")
    print("  Saved → submission_full_model.csv")

    # Summary
    print("\n" + "=" * 65)
    print("  SECONDARY OUTPUT SUMMARY")
    print("=" * 65)
    print(f"  Route benchmark OOF RMSE (no IDs) : {rmse_route:.2f} L")
    print(f"  Full model OOF RMSE (with IDs)     : {rmse_full:.2f} L")
    print(f"  Daily fleet Pearson corr           : {corr:.4f}")
    print(f"  Mean daily MAPE                    : {daily['pct_error'].mean():.1f}%")
    top3_ineff = delta_by_dump.head(3).index.tolist()
    top3_eff   = delta_by_dump.tail(3).index.tolist()
    print(f"  Most inefficient vehicles: {top3_ineff}")
    print(f"  Most efficient vehicles  : {top3_eff}")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    # Pass your TabPFN API token as first argument, e.g.:
    #   python tabpfn_model.py <TOKEN>
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python tabpfn_model.py <TABPFN_API_TOKEN>")
    token = sys.argv[1]
    run_tabpfn_submission(token)
    run_secondary_outputs(token)

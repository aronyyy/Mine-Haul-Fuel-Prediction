"""
Block 9 — LSTM sequence embedding features.

Trains a 2-layer LSTM (CPU, Huber loss) on raw preproc pings to predict
fuel_consumed_L.  After training, extracts the final hidden state of each
shift as a HIDDEN_DIM-dimensional embedding and a point fuel estimate.
These are merged into train_final / test_final.

Outputs (written to data/processed/):
  train_lstm_features.parquet
  test_lstm_features.parquet
  (also updates train_final.parquet and test_final.parquet in-place)
"""

import gc
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE, DROPOUT, EPOCHS, HIDDEN_DIM, LR, MAX_SEQ_LEN,
    N_FEATS, OUT_DIR, SEQ_FEATURES, load_fuel_labels,
)

warnings.filterwarnings("ignore")

KEY = ["dumper_id", "date_dpr", "shift"]


# ── Model ─────────────────────────────────────────────────────────────────────

class ShiftLSTM(nn.Module):
    def __init__(self, n_feats: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feats, hidden_size=hidden,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        last_hidden = hn[-1]                        # (B, hidden)
        pred        = self.head(last_hidden).squeeze(-1)  # (B,)
        return last_hidden, pred


def collate_batch(batch_seqs: list) -> tuple:
    """Pad a list of (T_i, F) arrays to (B, T_max, F) + lengths tensor."""
    lengths = torch.tensor([len(s) for s in batch_seqs], dtype=torch.long)
    T_max   = int(lengths.max())
    B, F    = len(batch_seqs), batch_seqs[0].shape[1]
    padded  = torch.zeros(B, T_max, F, dtype=torch.float32)
    for i, s in enumerate(batch_seqs):
        padded[i, :len(s)] = torch.from_numpy(s)
    return padded, lengths


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(
    preproc_path: str,
    fuel_labels: pd.DataFrame | None = None,
) -> tuple:
    """
    Convert a preproc parquet into per-shift sequence arrays.

    Returns
    -------
    seqs   : list of np.float32 arrays, each shape (T_i, N_FEATS)
    keys   : list of (dumper_id, date_str, shift) tuples
    labels : np.float32 array (NaN for test)
    """
    print(f"  Loading {preproc_path.split('/')[-1]} …")
    df = pd.read_parquet(preproc_path)
    df = df.sort_values(["vehicle", "ts"]).reset_index(drop=True)
    print(f"  Shape: {df.shape}")

    # Ping-level feature derivation
    gap   = df["is_gap_300s"].values == 1.0
    adiff = df.groupby("vehicle")["altitude"].diff().fillna(0).values.astype(np.float32)
    adiff[gap] = 0.0

    df["adiff_norm"] = (adiff / 10.0).astype(np.float32)
    df["log_dt"]     = np.log1p(df["dt_sec"].values).astype(np.float32)
    df["is_moving"]  = ((df["speed"] >= 2) & (df["ignition"] == 1)).astype(np.float32)
    df["spd_x_dt"]   = ((df["speed"].values * df["dt_sec"].values) / 3600).astype(np.float32)

    # Normalise to [0, 1] / symmetric range
    df["speed"]    = (df["speed"].values / 60.0).astype(np.float32)
    df["altitude"] = ((df["altitude"].values - 500) / 300.0).astype(np.float32)
    df["ignition"] = df["ignition"].astype(np.float32)
    df["gps_ok"]   = df["gps_ok"].astype(np.float32)

    # Build fuel lookup map
    fuel_map: dict = {}
    if fuel_labels is not None:
        for _, row in fuel_labels.iterrows():
            k = (row["dumper_id"], str(row["date_dpr"].date()), str(row["shift"]).strip())
            fuel_map[k] = row["fuel_consumed_L"]

    groups = df.groupby(["vehicle", "date_dpr", "shift_dpr"], observed=True, sort=False)
    seqs, keys, labels = [], [], []

    for (veh, date, shift), grp in groups:
        date_str  = str(pd.to_datetime(date).date()) if date != "nan" else "nan"
        shift_str = str(shift).strip()

        mat = grp[SEQ_FEATURES].values.astype(np.float32)

        # Evenly-spaced subsample if too long
        if len(mat) > MAX_SEQ_LEN:
            idx = np.linspace(0, len(mat) - 1, MAX_SEQ_LEN, dtype=int)
            mat = mat[idx]

        if len(mat) < 5:
            continue

        seqs.append(mat)
        keys.append((str(veh), date_str, shift_str))
        labels.append(fuel_map.get((str(veh), date_str, shift_str), np.nan))

    del df
    gc.collect()
    print(f"  Built {len(seqs):,} shift sequences")
    return seqs, keys, np.array(labels, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(
    model: ShiftLSTM,
    train_seqs: list,
    train_labels: np.ndarray,
) -> tuple:
    """Train the LSTM and return (model, fuel_mean, fuel_std)."""
    device    = torch.device("cpu")
    model.to(device)

    valid_idx = np.where(~np.isnan(train_labels) & (train_labels > 0))[0]
    print(f"  Training on {len(valid_idx):,} labelled shifts …")

    fuel_mean = float(np.nanmean(train_labels[valid_idx]))
    fuel_std  = float(np.nanstd(train_labels[valid_idx]))
    y_norm    = (train_labels[valid_idx] - fuel_mean) / (fuel_std + 1e-6)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=1.0)

    model.train()
    for epoch in range(EPOCHS):
        perm       = np.random.permutation(len(valid_idx))
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(perm), BATCH_SIZE):
            batch_pos = perm[start : start + BATCH_SIZE]
            batch_idx = valid_idx[batch_pos]

            x_pad, lengths = collate_batch([train_seqs[i] for i in batch_idx])
            y_batch        = torch.tensor(y_norm[batch_pos], dtype=torch.float32)

            optimizer.zero_grad()
            _, pred = model(x_pad.to(device), lengths)
            loss    = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}  loss={total_loss/n_batches:.4f}")

    return model, fuel_mean, fuel_std


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(
    model: ShiftLSTM,
    seqs: list,
    fuel_mean: float,
    fuel_std: float,
) -> tuple:
    """Return (embeddings, de-normalised predictions) as numpy arrays."""
    device = torch.device("cpu")
    model.eval()

    all_emb, all_pred = [], []
    with torch.no_grad():
        for start in range(0, len(seqs), BATCH_SIZE):
            x_pad, lengths = collate_batch(seqs[start : start + BATCH_SIZE])
            hidden, pred   = model(x_pad.to(device), lengths)
            all_emb.append(hidden.numpy())
            all_pred.append(pred.numpy())

    embeddings = np.vstack(all_emb).astype(np.float32)
    preds      = (np.concatenate(all_pred) * fuel_std + fuel_mean).astype(np.float32)
    return embeddings, preds


# ── Output assembly ───────────────────────────────────────────────────────────

def assemble_output(keys: list, embeddings: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    rows = []
    for (veh, date_str, shift), emb, pred in zip(keys, embeddings, preds):
        row = {
            "dumper_id"      : veh,
            "date_dpr"       : pd.to_datetime(date_str) if date_str != "nan" else pd.NaT,
            "shift"          : shift,
            "lstm_fuel_pred" : round(float(pred), 4),
        }
        for i, v in enumerate(emb):
            row[f"lstm_h{i:02d}"] = round(float(v), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_lstm_into_final(feat_path: str, final_path: str, tag: str) -> None:
    """Merge LSTM features into an existing *_final.parquet file."""
    final = pd.read_parquet(final_path)
    final["date_dpr"] = pd.to_datetime(final["date_dpr"])
    feats = pd.read_parquet(feat_path)

    lstm_cols = [c for c in feats.columns if c not in KEY]
    existing  = [c for c in lstm_cols if c in final.columns]
    if existing:
        final.drop(columns=existing, inplace=True)

    final = final.merge(feats, on=KEY, how="left")

    n_matched = final["lstm_h00"].notna().sum()
    print(f"  {tag}: {len(final):,} shifts | LSTM matched: {n_matched:,} ({100*n_matched/len(final):.1f}%)")

    n_miss = final["lstm_h00"].isna().sum()
    if n_miss > 0:
        print(f"    ⚠️  {n_miss} shifts unmatched — filling with 0")
        for col in lstm_cols:
            final[col] = final[col].fillna(0.0)

    final.to_parquet(final_path, index=False)
    print(f"  ✅  {tag} saved → {final_path.split('/')[-1]}  {final.shape}")
    del final, feats
    gc.collect()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_block9() -> None:
    print("=" * 70)
    print("BLOCK 9 — LSTM SEQUENCE EMBEDDING FEATURES")
    print("=" * 70)
    print(f"\n  Config: hidden={HIDDEN_DIM}, max_seq={MAX_SEQ_LEN}, "
          f"epochs={EPOCHS}, batch={BATCH_SIZE}")

    # Fuel labels
    print("\n  Loading fuel labels …")
    fuel_labels = load_fuel_labels()
    print(f"  Fuel records: {len(fuel_labels):,}")

    # Build sequences
    print("\n  ── TRAIN SEQUENCES ──")
    train_seqs, train_keys, train_labels = build_sequences(
        OUT_DIR + "train_dump_preproc.parquet", fuel_labels=fuel_labels
    )
    print("\n  ── TEST SEQUENCES ──")
    test_seqs, test_keys, _ = build_sequences(
        OUT_DIR + "test_dump_preproc.parquet", fuel_labels=None
    )

    # Train
    print("\n  ── TRAINING LSTM ──")
    model = ShiftLSTM(N_FEATS, HIDDEN_DIM, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    model, fuel_mean, fuel_std = train_lstm(model, train_seqs, train_labels)
    print(f"\n  Normalisation — mean={fuel_mean:.1f} L  std={fuel_std:.1f} L")

    # Extract embeddings
    print("\n  ── EXTRACTING TRAIN EMBEDDINGS ──")
    train_emb, train_preds = extract_embeddings(model, train_seqs, fuel_mean, fuel_std)
    print(f"  Train embeddings shape: {train_emb.shape}")
    valid = ~np.isnan(train_labels) & (train_labels > 0)
    if valid.sum() > 0:
        mae = np.abs(train_preds[valid] - train_labels[valid]).mean()
        print(f"  Train MAE (labelled shifts): {mae:.2f} L")

    print("\n  ── EXTRACTING TEST EMBEDDINGS ──")
    test_emb, test_preds = extract_embeddings(model, test_seqs, fuel_mean, fuel_std)
    print(f"  Test embeddings shape: {test_emb.shape}")

    # Assemble and save feature frames
    print("\n  ── ASSEMBLING OUTPUT ──")
    train_out = assemble_output(train_keys, train_emb, train_preds)
    test_out  = assemble_output(test_keys,  test_emb,  test_preds)

    train_out_path = OUT_DIR + "train_lstm_features.parquet"
    test_out_path  = OUT_DIR + "test_lstm_features.parquet"
    train_out.to_parquet(train_out_path, index=False)
    test_out.to_parquet(test_out_path,   index=False)
    print(f"  ✅  Saved → {train_out_path}")
    print(f"  ✅  Saved → {test_out_path}")

    # Merge into final files
    print("\n  ── MERGING INTO TRAIN_FINAL / TEST_FINAL ──")
    _merge_lstm_into_final(train_out_path, OUT_DIR + "train_final.parquet", "TRAIN")
    _merge_lstm_into_final(test_out_path,  OUT_DIR + "test_final.parquet",  "TEST")

    print("\n  BLOCK 9 COMPLETE ✅")
    print(f"  New columns: lstm_fuel_pred + lstm_h00 … lstm_h{HIDDEN_DIM-1:02d}")


if __name__ == "__main__":
    run_block9()

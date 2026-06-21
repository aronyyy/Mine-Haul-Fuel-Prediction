# Mine Haul Truck Fuel Prediction Pipeline

End-to-end feature engineering and modelling pipeline for the **MindShift Analytics Haul-Mark Challenge** (fuel consumption prediction for mine haul trucks).

## Project Structure

```
fuel_pipeline/
├── main.py                  ← orchestrator (run full pipeline or single stages)
├── src/
│   ├── config.py            ← all constants, paths, shared helpers
│   ├── preprocess.py        ← Blocks 1 & 2  – raw telemetry → preproc parquets
│   ├── feature_engineering.py  ← Blocks 3 & 6  – shift-level feature aggregation
│   ├── trip_detection.py    ← Blocks 4 & 7a – loading/dump event state machine
│   ├── assemble_final.py    ← Blocks 5 & 7b – merge features + interaction terms
│   ├── lstm_embeddings.py   ← Block 9        – LSTM sequence embeddings
│   └── tabpfn_model.py      ← TabPFN 5-fold CV + secondary outputs
├── data/
│   ├── raw/                 ← all competition inputs (parquets, CSVs, geojson)
│   └── processed/           ← all pipeline outputs (parquets, submission CSVs)
└── requirements.txt
```

## Data Layout (`data/raw/`)

| File | Description |
|------|-------------|
| `telemetry_YYYY-MM-DD_YYYY-MM-DD.parquet` | Raw GPS/sensor pings |
| `smry_jan/feb/mar_train_ordered.csv` | Shift-level fuel summary (train) |
| `fleet.csv` | Vehicle metadata (mine, dump_switch flag) |
| `id_mapping_new.csv` | Test shift → submission ID mapping |
| `mine001_dump_zones_3d.geojson` | Geofenced dump zones for mine001 |

## Pipeline Stages

| Stage | Script | Outputs |
|-------|--------|---------|
| `preprocess` | `preprocess.py` | `train/test_dump_preproc.parquet`, `train/test_loaders_preproc.parquet` |
| `features` | `feature_engineering.py` | `train/test_features.parquet` |
| `trips` | `trip_detection.py` | `train/test_trip_features.parquet` |
| `assemble` | `assemble_final.py` | `train/test_final.parquet` |
| `lstm` | `lstm_embeddings.py` | `train/test_lstm_features.parquet` (also updates final parquets) |
| `model` | `tabpfn_model.py` | submission CSVs, `efficiency_delta.csv`, charts |

## Usage

```bash
# Full pipeline (model stage requires --token)
python main.py --token <TABPFN_API_TOKEN>

# Single stage
python main.py --stage preprocess
python main.py --stage features
python main.py --stage trips
python main.py --stage assemble
python main.py --stage lstm
python main.py --stage model --token <TABPFN_API_TOKEN>

# Or run individual scripts directly
cd src
python preprocess.py
python feature_engineering.py
python trip_detection.py
python assemble_final.py
python lstm_embeddings.py
python tabpfn_model.py <TABPFN_API_TOKEN>
```

## Dependencies

```
pandas
numpy
pyproj
geopandas          
torch
lightgbm
tabpfn-client
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install -r requirements.txt
```

## Key Design Decisions

- **Dual distance**: both `speed×time` and haversine-based km are kept as separate features so the model can choose the more reliable signal per vehicle/mine.
- **Gap classification**: data gaps ≥300 s are labelled `engine_off_stationary` or `engine_running_blind` based on fuel volume change, rather than discarded.
- **Trip state machine**: a single forward pass classifies every ping as loaded/empty; no resampling required.
- **LSTM irregular intervals**: `log1p(dt_sec)` is an explicit model input so the network learns to weight each observation by its temporal coverage.
- **TabPFN feature selection**: a LightGBM scout ranks physical features; the top 20 are taken, then **all** LSTM embedding columns are force-included regardless of scout ranking.

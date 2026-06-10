"""
Main pipeline runner.

Runs all stages in order.  Pass --stage <name> to run a single stage,
or run with no arguments to execute the full pipeline end-to-end.

Stages
------
  preprocess   — Blocks 1 & 2  (raw telemetry → preproc parquets)
  features     — Blocks 3 & 6  (preproc → shift-level features)
  trips        — Blocks 4 & 7a (trip detection)
  assemble     — Blocks 5 & 7b (merge + interaction features → final parquets)
  lstm         — Block 9        (LSTM embeddings → update final parquets)
  model        — TabPFN train + submission export

Usage
-----
  python main.py                                # full pipeline
  python main.py --stage preprocess             # single stage
  python main.py --stage model --token <TOKEN>  # TabPFN needs API token
"""

import argparse
import os
import sys

from config import OUT_DIR


def ensure_output_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def stage_preprocess() -> None:
    from preprocess import run_preprocess
    from config import TRAIN_FILES, TEST_FILES
    run_preprocess(TRAIN_FILES, tag="train", is_train=True)
    run_preprocess(TEST_FILES,  tag="test",  is_train=False)


def stage_features() -> None:
    from feature_engineering import build_shift_features
    build_shift_features(
        OUT_DIR + "train_dump_preproc.parquet",
        OUT_DIR + "train_features.parquet",
        is_train=True,
    )
    build_shift_features(
        OUT_DIR + "test_dump_preproc.parquet",
        OUT_DIR + "test_features.parquet",
        is_train=False,
    )


def stage_trips() -> None:
    from trip_detection import detect_trips
    detect_trips(
        OUT_DIR + "train_dump_preproc.parquet",
        OUT_DIR + "train_loaders_preproc.parquet",
        OUT_DIR + "train_features.parquet",
        OUT_DIR + "train_trip_features.parquet",
    )
    detect_trips(
        OUT_DIR + "test_dump_preproc.parquet",
        OUT_DIR + "test_loaders_preproc.parquet",
        OUT_DIR + "test_features.parquet",
        OUT_DIR + "test_trip_features.parquet",
    )


def stage_assemble() -> None:
    from assemble_final import assemble_train, assemble_test
    assemble_train()
    assemble_test()


def stage_lstm() -> None:
    from lstm_embeddings import run_block9
    run_block9()


def stage_model(token: str) -> None:
    from tabpfn_model import run_tabpfn_submission, run_secondary_outputs
    run_tabpfn_submission(token)
    run_secondary_outputs(token)


STAGES = {
    "preprocess": stage_preprocess,
    "features":   stage_features,
    "trips":      stage_trips,
    "assemble":   stage_assemble,
    "lstm":       stage_lstm,
    "model":      stage_model,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine haul truck fuel pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default=None,
        help="Run a single stage (default: all stages in order)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="TabPFN API token (required for --stage model)",
    )
    args = parser.parse_args()

    ensure_output_dir()

    if args.stage is not None:
        fn = STAGES[args.stage]
        if args.stage == "model":
            if not args.token:
                sys.exit("Error: --token is required for --stage model")
            fn(args.token)
        else:
            fn()
    else:
        # Full pipeline
        for name, fn in STAGES.items():
            print(f"\n{'='*70}")
            print(f"  STAGE: {name.upper()}")
            print(f"{'='*70}")
            if name == "model":
                if not args.token:
                    print("  Skipping model stage — no --token provided")
                    continue
                fn(args.token)
            else:
                fn()
        print("\n🎉  FULL PIPELINE COMPLETE")


if __name__ == "__main__":
    main()

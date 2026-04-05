from pathlib import Path
import time

import pandas as pd

from valentine import valentine_match
from valentine.algorithms import Coma, JaccardDistanceMatcher, DistributionBased, GoodnessOfFit
from valentine.metrics import (
    F1Score,
    MissingAccuracy,
    NewAccuracy,
    PersistentAccuracy,
    Precision,
    PrecisionTopNPercent,
    Recall,
    RecallAtSizeofGroundTruth,
)

DATA_DIR = Path("/home/exati/Facul/tcc/datasets/simcaq/escola_cleaned")
OUTPUT_DIR = Path("/home/exati/Facul/tcc/results")
START_YEAR = 2007
END_YEAR = 2023  # inclusive; compares END_YEAR with END_YEAR+1


def process_year_pair(year: int, matcher, sample_size: int) -> dict:
    year_next = year + 1
    algo_name = type(matcher).__name__

    d1_path = DATA_DIR / f"only_num_escola_{year}_cleaned.csv"
    d2_path = DATA_DIR / f"only_num_escola_{year_next}_cleaned.csv"

    print(f"[{year} -> {year_next}] Loading data...")
    df1 = pd.read_csv(d1_path, low_memory=False).sample(n=sample_size, random_state=42)
    df2 = pd.read_csv(d2_path, low_memory=False).sample(n=sample_size, random_state=42)

    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)

    persisted_columns = df1_columns & df2_columns
    missing_columns = df1_columns - df2_columns   # in source but not target
    new_columns = df2_columns - df1_columns       # in target but not source

    ground_truth = [(col, col) for col in persisted_columns]

    print(f"[{year} -> {year_next}] Running matcher ({algo_name})...")
    max_size = max(len(df1), len(df2))
    start_time = time.time()
    matches = valentine_match([df1, df2], matcher, instance_sample_size=max_size).take_top_n_per_source(10)
    elapsed = time.time() - start_time
    print(f"[{year} -> {year_next}] Matching time: {elapsed:.2f}s | pairs found: {len(matches)}")

    # --- Standard metrics -------------------------------------------------
    core_metrics = matches.get_metrics(
        ground_truth,
        metrics={
            Precision(),
            F1Score(),
            Recall(),
            PrecisionTopNPercent(n=10),
            RecallAtSizeofGroundTruth(),
        },
    )

    # --- Schema-change metrics --------------------------------------------
    special_metrics = matches.get_metrics(
        ground_truth,
        metrics={
            PersistentAccuracy(),
            MissingAccuracy(source_columns=tuple(df1.columns)),
            NewAccuracy(target_columns=tuple(df2.columns)),
        },
    )

    # --- Build results row ------------------------------------------------
    row = {
        "Algorithm": algo_name,
        "Source year": year,
        "Target year": year + 1,
        "Matching time (s)": round(elapsed, 2),
        "Total columns": len(persisted_columns) + len(new_columns),
        "Persisted columns": len(persisted_columns),
        "New columns": len(new_columns),
        "Missing columns": len(missing_columns),
        "Precision": core_metrics.get("Precision"),
        "F1": core_metrics.get("F1Score"),
        "Recall": core_metrics.get("Recall"),
        "PrecisionTop10": core_metrics.get("PrecisionTop10Percent"),
        "RecallAtSizeofGroundTruth": core_metrics.get("RecallAtSizeofGroundTruth"),
        "PersistentAccuracy": special_metrics.get("PersistentAccuracy"),
        "NewAccuracy": special_metrics.get("NewAccuracy"),
        "MissingAccuracy": special_metrics.get("MissingAccuracy"),
    }

    print(f'row for {year} -> {year_next}: {row}')
    print('---' * 40)

    # --- Build matches dataframe ------------------------------------------
    matches_rows = [
        {
            "source_column": pair.source_column,
            "target_column": pair.target_column,
            "statistic": score,
        }
        for pair, score in matches.items()
    ]
    matches_df = pd.DataFrame(matches_rows)
    matches_filename = OUTPUT_DIR / f"matches_escola_{sample_size}_{year}_{year_next}_{algo_name}.csv"
    matches_df.to_csv(matches_filename, index=False)
    print(f"[{year} -> {year_next}] Saved matches  -> {matches_filename}")

    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matcher = GoodnessOfFit(hist_bin=10)
    matcher = JaccardDistanceMatcher()
    matcher = DistributionBased()
    algo_name = type(matcher).__name__

    sample_sizes = [100, 500, 1000, 5000, 10000, 20000]
    for sample_size in sample_sizes:
        results_filename = OUTPUT_DIR / f"resultado_escola_{sample_size}_{START_YEAR}_{END_YEAR + 1}_{algo_name}.csv"
        # Remove existing file so we start fresh
        results_filename.unlink(missing_ok=True)
        for year in range(START_YEAR, END_YEAR + 1):
            row = process_year_pair(year, matcher, sample_size)
            write_header = not results_filename.exists()
            pd.DataFrame([row]).to_csv(results_filename, mode="a", header=write_header, index=False)
            print(f"[{year}] Appended metrics -> {results_filename}")

    print(f"\nSaved all metrics -> {results_filename}")
    print("\nDone.")


if __name__ == "__main__":
    main()

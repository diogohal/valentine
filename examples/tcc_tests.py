import argparse
from pathlib import Path
import time

import pandas as pd
import random
import string

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
	MeanReciprocalRank
	
)

DATA_DIR = Path("/home/exati/Facul/tcc/datasets/escola_agregado/cleaned")
OUTPUT_DIR = Path("/home/exati/Facul/tcc/results/escola_agregada")
START_YEAR = 2007
END_YEAR = 2023  # inclusive; compares END_YEAR with END_YEAR+1

def build_random_column_name(existing_names, size=5):
	while True:
		candidate = "".join(random.choices(string.ascii_uppercase, k=size))
		if candidate not in existing_names:
			return candidate


def build_random_3digit_suffix_column_name(base_name, existing_names):
	while True:
		candidate = f"{base_name}{random.randint(0, 999):03d}"
		if candidate not in existing_names:
			return candidate


def transform_target_column_names(
	df_target,
	ground_truth,
	persisted_columns,
	new_columns=None,
	transform_strategy="reverse",
):
	if transform_strategy == "none":
		column_mapping = {column: column for column in df_target.columns}
	elif transform_strategy == "reverse":
		column_mapping = {column: column[::-1] for column in df_target.columns}
	elif transform_strategy == "random5":
		used_names = set(df_target.columns)
		column_mapping = {}
		for column in df_target.columns:
			random_name = build_random_column_name(used_names)
			column_mapping[column] = random_name
			used_names.add(random_name)
	elif transform_strategy == "suffix_random3":
		used_names = set(df_target.columns)
		column_mapping = {}
		for column in df_target.columns:
			random_name = build_random_3digit_suffix_column_name(column, used_names)
			column_mapping[column] = random_name
			used_names.add(random_name)
	else:
		raise ValueError(
			f"Unknown transform strategy: {transform_strategy}. "
			"Use one of: none, reverse, random5, suffix_random3."
		)

	df_target = df_target.rename(columns=column_mapping)

	transformed_ground_truth = [
		(source_column, column_mapping.get(target_column, target_column))
		for source_column, target_column in ground_truth
	]

	transformed_persisted_columns = {
		column_mapping.get(column, column) for column in persisted_columns
	}

	transformed_new_columns = {
		column_mapping.get(column, column) for column in (new_columns or set())
	}

	return df_target, transformed_ground_truth, transformed_persisted_columns, transformed_new_columns


def process_year_pair(year: int, matcher, sample_size: int = None, args=None) -> dict:
    year_next = year + 1
    algo_name = type(matcher).__name__

    d1_path = DATA_DIR / f"dados_final_{year}.csv"
    d2_path = DATA_DIR / f"dados_final_{year_next}.csv"

    print(f"[{year} -> {year_next}] Loading data...")

    df1 = pd.read_csv(d1_path, low_memory=False)
    df2 = pd.read_csv(d2_path, low_memory=False)
    if sample_size:
        df1 = df1.sample(n=sample_size, random_state=42)
        df2 = df2.sample(n=sample_size, random_state=42)

    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)

    persisted_columns = df1_columns & df2_columns
    missing_columns = df1_columns - df2_columns   # in source but not target
    new_columns = df2_columns - df1_columns       # in target but not source

    ground_truth = [(col, col) for col in persisted_columns]

    if args and args.target_column_transform != "none":
        print(f'Transforming target column names using strategy: {args.target_column_transform}')
        df2, ground_truth, persisted_columns, new_columns = transform_target_column_names(
            df2, ground_truth, persisted_columns,
            new_columns=new_columns,
            transform_strategy=args.target_column_transform,
        )

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
			MeanReciprocalRank()
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
		"MeanReciprocalRank": core_metrics.get("MeanReciprocalRank"),
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
    parser = argparse.ArgumentParser(
		description="Run Magneto benchmark on SimCAQ Escola year-pair tables"
	)
    parser.add_argument(
		"--algorithm",
		type=str,
		default="jaccard",
		choices=["jaccard", "goodnessoffit", "coma"],
		help="Matching algorithm to use.",
	)
    parser.add_argument(
		"--target_column_transform",
		type=str,
		default="none",
		choices=["none", "reverse", "random5", "suffix_random3"],
		help="Transformation strategy for target column names.",
	)
	
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.algorithm == "coma":
        matcher = Coma(use_instances=True, use_schema=False)
    elif args.algorithm == "goodnessoffit":
        matcher = GoodnessOfFit(hist_bin=10)
    else:
        matcher = JaccardDistanceMatcher()

    algo_name = type(matcher).__name__

    sample_sizes = [None]
    for sample_size in sample_sizes:
        results_filename = OUTPUT_DIR / f"resultado_escola_total_{START_YEAR}_{END_YEAR + 1}_{algo_name}.csv"
        # Remove existing file so we start fresh
        results_filename.unlink(missing_ok=True)
        for year in range(START_YEAR, END_YEAR + 1):
            row = process_year_pair(year=year, matcher=matcher, sample_size=sample_size, args=args)
            write_header = not results_filename.exists()
            pd.DataFrame([row]).to_csv(results_filename, mode="a", header=write_header, index=False)
            print(f"[{year}] Appended metrics -> {results_filename}")

    print(f"\nSaved all metrics -> {results_filename}")
    print("\nDone.")


if __name__ == "__main__":
    main()

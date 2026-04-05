import pprint
from pathlib import Path

import pandas as pd

from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher
from valentine.algorithms import GoodnessOfFit
from valentine.metrics import F1Score, PrecisionTopNPercent, PersistentAccuracy, MissingAccuracy, NewAccuracy

pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)


def main():
    # Load data using pandas
    d1_path = Path("data") / "abalone1.csv"
    d2_path = Path("data") / "abalone2.csv"
    df1 = pd.read_csv(d1_path)
    df2 = pd.read_csv(d2_path)

    print(f'Columns in df1: {df1.columns}')
    print(f'Columns in df2: {df2.columns}')

    # Instantiate matcher and run
    matcher = JaccardDistanceMatcher()
    matcher = GoodnessOfFit(continuous_threshold=20, p_value_threshold=0.95)
    matches = valentine_match(df1, df2, matcher)

    # MatcherResults is a wrapper object that has several useful
    # utility/transformation functions
    print("Found the following matches:")
    pp.pprint(matches)

    print("\nGetting the one-to-one matches:")
    matches = matches.take_top_n_per_source(2)
    pp.pprint(matches)

    # If ground truth available valentine could calculate the metrics
    ground_truth = [("A", "A"), ("E", "E"), ("H", "H"), ("F", "F"), ("B", "B"), ("G", "G")]

    metrics = matches.get_metrics(ground_truth)

    print("\nAccording to the ground truth:")
    pp.pprint(ground_truth)

    print("\nThese are the scores of the default metrics for the matcher:")
    pp.pprint(metrics)

    print("\nSpecific metrics:")
    pp.pprint(matches.get_metrics(ground_truth, metrics={PersistentAccuracy(), MissingAccuracy(source_columns=tuple(df1.columns)), NewAccuracy(target_columns=tuple(df2.columns))}))

    print("\nThe MatcherResults object is a dict and can be treated such:")
    for match in matches:
        print(f"{match!s: <60} {matches[match]}")


if __name__ == "__main__":
    main()

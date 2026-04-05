import pprint
from pathlib import Path

import pandas as pd

from valentine import valentine_match
from valentine.algorithms import Coma, GoodnessOfFit
from valentine.metrics import F1Score, PrecisionTopNPercent, PersistentAccuracy, MissingAccuracy, NewAccuracy

pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)


def main():
    d1_path = '/home/exati/Facul/tcc/datasets/gdc/source-tables-prepared/prepared_Cao.csv'
    d2_path = '/home/exati/Facul/tcc/datasets/gdc/target-tables-prepared/gdc_unique_columns_concat_values_prepared.csv'
    ground_truth_path = '/home/exati/Facul/tcc/datasets/gdc/ground-truth/Cao.csv'

    df1 = pd.read_csv(d1_path, low_memory=False)
    df2 = pd.read_csv(d2_path, low_memory=False)
    ground_truth_df = pd.read_csv(ground_truth_path, low_memory=False)

    df1_columns = df1.columns
    filtered_gt = ground_truth_df[ground_truth_df['original_paper_variable_names'].isin(df1.columns)]

    matcher = Coma(use_instances=True, use_schema=False)
    matcher = GoodnessOfFit(continuous_threshold=len(df1) * 0.01)
    max_size = max(len(df1), len(df2))
    matches = valentine_match([df1, df2], matcher, instance_sample_size=max_size).take_top_n_per_source(3)

    ground_truth = []
    for _, row in filtered_gt.iterrows():
        source_column = row['original_paper_variable_names']
        target_column = row['GDC_format_variable_names']
        ground_truth.append((source_column, target_column))

    metrics = matches.get_metrics(ground_truth)

    print("\nThese are the scores of the default metrics for the matcher:")
    pp.pprint(metrics)

    special_metrics = matches.get_metrics(
        ground_truth,
        metrics={
            PersistentAccuracy(),
            MissingAccuracy(source_columns=tuple(df1.columns)),
            NewAccuracy(target_columns=tuple(df2.columns)),
        },
    )
    print("\nThese are the scores of the special metrics for the matcher:")
    pp.pprint(special_metrics)

if __name__ == "__main__":
    main()

from typing import List
import numpy as np
from multiprocessing import shared_memory
import multiprocessing as mp
import math

from ..base_matcher import BaseMatcher
from ..match import Match
from ...data_sources.base_table import BaseTable
from .gof_methods import ks_test, ad_test, chisq_test, g_test


class GoodnessOfFit(BaseMatcher):
    """Goodness-of-Fit matcher.

    Parameters
    ----------
    top_ranking: int
        Number of top matches to return (stubbed, not used yet).
    continuous_threshold: int
        Threshold for continuous data (stubbed, not used yet).
    p_value_threshold: float
        P-value threshold (stubbed, not used yet).
    """

    def __init__(self, top_ranking: int = 10, continuous_threshold: int = 127, p_value_threshold: float = 0.95, hist_bin: int = 10):
        self.top_ranking = int(top_ranking)
        self.continuous_threshold = int(continuous_threshold)
        self.p_value_threshold = float(p_value_threshold)
        self.hist_bin = int(hist_bin)
        self.__target_name: str = ""

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> dict:
        """
        Overridden function of BaseMatcher that takes the source and target tables
        and returns a ranked dict of column pair matches.

        Returns
        -------
        dict
            A dictionary with matches keyed by ((src_table, src_col), (tgt_table, tgt_col))
            and similarity as value.
        """
        self.__target_name = target_input.name

        t1 = source_input
        t2 = target_input

        try:
            base_df = t1.get_instances_df()
            new_df = t2.get_instances_df()
            base_cols = list(base_df.columns)
            new_cols = list(new_df.columns)
            base_data = base_df.to_numpy()
            new_data = new_df.to_numpy()
        except Exception as e:
            print(f"Error converting columns to numpy arrays: {e}. Skipping matching.")
            return {}

        results = self.parallel_cart_prod(
            base_cols, base_data, new_cols, new_data,
            t1.name, t2.name,
            delimiter=self.continuous_threshold, hist_bin=self.hist_bin
        )

        filtered_results = self.getOnlyHighestPValueBetweenTests(results)
        filtered_results = self.truncateResultsForEachColumn(filtered_results)
        return self.__to_matches(filtered_results, t1, t2)

    def __to_matches(self, results: list, t1: BaseTable, t2: BaseTable) -> dict:
        """Convert internal result rows to a Valentine matches dict.

        For each column pair the highest p-value among its tests is used as
        the similarity score, consistent with how other matchers return a
        single similarity per pair.
        """
        # Aggregate: keep the best p-value for each (col1, col2) pair
        best: dict[tuple, float] = {}
        for result in results:
            key = (result[2], result[3])  # (col1, col2)
            pvalue = float(result[6])
            if key not in best or pvalue > best[key]:
                best[key] = pvalue

        matches = {}
        for (col1, col2), sim in best.items():
            if self.__target_name == t1.name:
                matches.update(Match(t1.name, col1, t2.name, col2, sim).to_dict)
            else:
                matches.update(Match(t2.name, col2, t1.name, col1, sim).to_dict)
        return matches
    
    def attr_cart_product(self, base_cols, base_data, new_cols, new_data, dist1, dist2, delimiter=127, hist_bin=50):
        results = []
        for i, col1 in enumerate(base_cols):
            for j, col2 in enumerate(new_cols):
                nuniq1 = len(np.unique(base_data[:, i]))
                nuniq2 = len(np.unique(new_data[:, j]))

                if nuniq1 < 2:
                    break
                elif nuniq2 < 2:
                    continue

                data1 = base_data[:, i]
                data2 = new_data[:, j]

                try:
                    data1 = data1.astype(float)
                    data2 = data2.astype(float)
                except ValueError:
                    print(f"Skipping comparison due to non-numeric data.")
                    continue      
                if nuniq1 <= delimiter and nuniq2 <= delimiter:
                    uniq1 = np.unique(data1)
                    uniq2 = np.unique(data2)

                    if len(np.intersect1d(uniq1, uniq2)) < 2:
                        continue
                    res = chisq_test(data1, data2)
                    results.append([dist1, dist2, col1, col2, 'CHISQ', res.statistic, res.pvalue])
                    res = g_test(data1, data2)
                    results.append([dist1, dist2, col1, col2, 'G', res.statistic, res.pvalue])

                elif nuniq1 > delimiter and nuniq2 > delimiter:
                    res = ks_test(data1, data2, hist_bin)
                    results.append([dist1, dist2, col1, col2, 'KS', res.statistic, res.pvalue])
                    res = ad_test(data1, data2, hist_bin)
                    results.append([dist1, dist2, col1, col2, 'AD', res.statistic, res.pvalue])
        return results
    
    def getOnlyHighestPValueBetweenTests(self, results):
        # Group results by col1 and col2
        grouped = {}
        for result in results:
            key = (result[2], result[3])  # (col1, col2)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        # Keep all results in a pair if any test in that pair exceeds the threshold.
        filtered_results = []
        for key, group in grouped.items():
            if any(r[6] >= self.p_value_threshold for r in group):
                filtered_results.extend(group)

        return filtered_results
    
    def truncateResultsForEachColumn(self, results):
        # Group by col1
        grouped_by_col1 = {}
        for result in results:
            col1 = result[2]
            if col1 not in grouped_by_col1:
                grouped_by_col1[col1] = []
            grouped_by_col1[col1].append(result)

        truncated_results = []
        for col1, col1_results in grouped_by_col1.items():
            # For this col1, group by (col1, col2)
            grouped_pairs = {}
            for result in col1_results:
                key = (result[2], result[3])  # (col1, col2)
                if key not in grouped_pairs:
                    grouped_pairs[key] = []
                grouped_pairs[key].append(result)

            # For each pair, get max p-value
            pair_max_p = []
            for key, group in grouped_pairs.items():
                max_p = max(r[6] for r in group)
                pair_max_p.append((key, max_p, group))

            # Sort by max p-value descending
            pair_max_p.sort(key=lambda x: x[1], reverse=True)

            # Take top_ranking
            top_pairs = pair_max_p[:self.top_ranking]

            # Add all results from top pairs
            for key, max_p, group in top_pairs:
                truncated_results.extend(group)

        return truncated_results


    # ---------- Parallel matching functions ----------
    @staticmethod
    def worker_compare(base_cols, shm_base, new_cols, shm_new, dtype_base, dtype_new, 
                    base_shape, new_shape, start, stop, dist1, dist2, 
                    delimiter=127, hist_bin=10, result_list=None):

        base_arr = np.ndarray(base_shape, dtype=dtype_base, buffer=shm_base.buf)
        new_arr = np.ndarray(new_shape, dtype=dtype_new, buffer=shm_new.buf)

        results = []

        for i in range(start, stop):
            for j in range(len(new_cols)):
                col1 = base_cols[i]
                col2 = new_cols[j]

                try:
                    nuniq1 = len(np.unique(base_arr[:, i]))
                    nuniq2 = len(np.unique(new_arr[:, j]))
                except ValueError:
                    continue

                if nuniq1 < 2:
                    break
                elif nuniq2 < 2:
                    continue

                if nuniq1 <= delimiter and nuniq2 <= delimiter:
                    uniq1 = np.unique(base_arr[:, i])
                    uniq2 = np.unique(new_arr[:, j])

                    if len(np.intersect1d(uniq1, uniq2)) < 2:
                        continue
                    res = chisq_test(base_arr[:, i], new_arr[:, j])
                    results.append([dist1, dist2, col1, col2, 'CHISQ', res.statistic, res.pvalue])
                    res = g_test(base_arr[:, i], new_arr[:, j])
                    results.append([dist1, dist2, col1, col2, 'G', res.statistic, res.pvalue])
                elif nuniq1 > delimiter and nuniq2 > delimiter:
                    res = ks_test(base_arr[:, i], new_arr[:, j], hist_bin)
                    results.append([dist1, dist2, col1, col2, 'KS', res.statistic, res.pvalue])
                    res = ad_test(base_arr[:, i], new_arr[:, j], hist_bin)
                    results.append([dist1, dist2, col1, col2, 'AD', res.statistic, res.pvalue])
        if result_list is not None:
            result_list.extend(results)


    @staticmethod
    def parallel_cart_prod(base_cols, base_data, new_cols, new_data,
                        dist1, dist2, delimiter=127, hist_bin=10, num_workers=None):
        # Celery worker processes are daemonized, so they cannot spawn child processes.
        # Fall back to the sequential version when running in a daemon.
        if mp.current_process().daemon:
            print("[parallel_cart_prod] Running sequential fallback because current process is daemonic.")
            return GoodnessOfFit().attr_cart_product(base_cols, base_data, new_cols, new_data, dist1, dist2, delimiter, hist_bin)

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 4)

        print(f"Starting parallel cartesian product with {num_workers} workers...")

        # Start and copy input shared memory
        shm_base = shared_memory.SharedMemory(create=True, size=base_data.nbytes)
        shm_new = shared_memory.SharedMemory(create=True, size=new_data.nbytes)
        base_sh = np.ndarray(base_data.shape, dtype=base_data.dtype, buffer=shm_base.buf)
        new_sh = np.ndarray(new_data.shape, dtype=new_data.dtype, buffer=shm_new.buf)
        base_sh[:] = base_data[:]
        new_sh[:] = new_data[:]

        # Divide base_data in chunks for each process
        n = len(base_cols)
        chunk = math.ceil(n / num_workers)

        manager = mp.Manager()
        result_list = manager.list()

        processes = []
        for id in range(num_workers):
            start = id * chunk
            if start >= n:
                break
            stop = min((id+1) * chunk, n)
            p = mp.Process(target=GoodnessOfFit.worker_compare,
                        args=(base_cols, shm_base, new_cols, shm_new, base_data.dtype, 
                                new_data.dtype, base_data.shape, new_data.shape, 
                                start, stop, dist1, dist2, delimiter, hist_bin, result_list))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Cleanup shared memory
        shm_base.close()
        shm_base.unlink()
        shm_new.close()
        shm_new.unlink()

        return list(result_list)

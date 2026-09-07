"""
Distribution statistics, a paired nonparametric significance test, and a
cross-client fairness summary for PMCS federated evaluation.

Standalone: only numpy and scipy.stats are used (no pandas, no pydantic).
Every function here is pure and operates on plain dicts/lists/floats so its
output is directly JSON-serialisable.
"""
import math

import numpy as np
from scipy import stats


def distribution_stats(values):
    """Summary statistics over per-case metric values.

    `values` is any iterable of floats that may contain None entries (e.g. an
    undefined sensitivity/specificity for an all-background case, or an HD95
    excluded on prediction/reference mismatch). None entries are dropped
    before computing anything -- they are not treated as 0 and not treated as
    NaN, since either of those would silently distort mean/std/median.

    If nothing is left after dropping None, there is no defined distribution:
    every stat except n_cases is returned as None (not NaN) so downstream
    JSON/aggregation sees an explicit "no data" rather than a poisoning NaN.
    """
    filtered = [float(v) for v in values if v is not None]
    n_cases = len(filtered)
    if n_cases == 0:
        return {
            'mean': None, 'std': None, 'median': None,
            'iqr_25': None, 'iqr_75': None,
            'min': None, 'max': None, 'n_cases': 0,
        }
    arr = np.asarray(filtered, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=0)),
        'median': float(np.median(arr)),
        'iqr_25': float(np.percentile(arr, 25)),
        'iqr_75': float(np.percentile(arr, 75)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'n_cases': n_cases,
    }


def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank test on paired per-case values (e.g. the same
    cases' Dice under two configurations), with an approximate effect size.

    Pairs where either side is None are dropped first (only cases present in
    both configurations are compared). If fewer than 2 valid pairs remain, or
    every remaining pair is identical (a-b == 0 everywhere -- the case where
    scipy.stats.wilcoxon itself raises because the zero-difference default
    zero_method has nothing left to rank), this returns a graceful degenerate
    result instead of raising: statistic/p_value/effect_size_r are None,
    n_pairs is the count of valid pairs found, and 'note' says why.

    No significance verdict is computed or returned here -- only the raw
    numbers; the caller decides what counts as significant.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (paired per-case values)")

    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n_pairs = len(pairs)

    def degenerate(note):
        return {'statistic': None, 'p_value': None, 'effect_size_r': None,
                'n_pairs': n_pairs, 'note': note}

    if n_pairs < 2:
        return degenerate('insufficient pairs')

    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    diffs = xs - ys

    if np.all(diffs == 0):
        return degenerate('identical')

    try:
        statistic, p_value = stats.wilcoxon(xs, ys)
    except ValueError as e:
        return degenerate('wilcoxon failed: %s' % e)

    median_diff = float(np.median(diffs))
    if median_diff > 0:
        sign = 1
    elif median_diff < 0:
        sign = -1
    else:
        sign = 0

    z_magnitude = stats.norm.isf(p_value / 2.0)
    effect_size_r = sign * z_magnitude / math.sqrt(n_pairs)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'effect_size_r': float(effect_size_r),
        'n_pairs': n_pairs,
    }


def fairness_block(per_client_scores, client_modalities, modality_names=('FLAIR', 'T1ce', 'T1', 'T2')):
    """Cross-client fairness summary for one round/partition of scores.

    per_client_scores: {client_idx: float}, e.g. each client's mean Dice.
    client_modalities: {client_idx: [bool, bool, bool, bool]}, same client's
        modality mask in `modality_names` order.
    """
    client_ids = list(per_client_scores.keys())
    scores = np.array([per_client_scores[c] for c in client_ids], dtype=float)

    worst_pos = int(np.argmin(scores))
    best_pos = int(np.argmax(scores))
    worst_client = client_ids[worst_pos]
    best_client = client_ids[best_pos]
    worst_score = float(scores[worst_pos])
    best_score = float(scores[best_pos])

    cutoff = np.percentile(scores, 25)
    worst_mask = scores <= cutoff
    if not np.any(worst_mask):
        # Degenerate case (shouldn't happen with <=, since the minimum always
        # satisfies it, but guarded per spec): fall back to the single worst.
        worst_quartile_mean = worst_score
    else:
        worst_quartile_mean = float(np.mean(scores[worst_mask]))

    per_modality = {}
    for i, name in enumerate(modality_names):
        holders = [c for c in client_ids if client_modalities[c][i]]
        if holders:
            vals = [per_client_scores[c] for c in holders]
            per_modality[name] = {'mean_score': float(np.mean(vals)), 'n_clients': len(holders)}
        else:
            per_modality[name] = {'mean_score': None, 'n_clients': 0}

    return {
        'worst_client': worst_client,
        'best_client': best_client,
        'worst_client_score': worst_score,
        'best_client_score': best_score,
        'std_across_clients': float(np.std(scores, ddof=0)),
        'range': float(np.max(scores) - np.min(scores)),
        'worst_quartile_mean': worst_quartile_mean,
        'per_modality_subset_aggregates': per_modality,
    }


if __name__ == '__main__':
    # --- distribution_stats -------------------------------------------------
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, None]
    d = distribution_stats(vals)
    filtered_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert d['n_cases'] == 5
    assert math.isclose(d['mean'], float(np.mean(filtered_np)), rel_tol=1e-9)
    assert math.isclose(d['std'], float(np.std(filtered_np, ddof=0)), rel_tol=1e-9)
    assert math.isclose(d['median'], float(np.median(filtered_np)), rel_tol=1e-9)
    assert math.isclose(d['min'], float(np.min(filtered_np)), rel_tol=1e-9)
    assert math.isclose(d['max'], float(np.max(filtered_np)), rel_tol=1e-9)
    assert math.isclose(d['iqr_25'], float(np.percentile(filtered_np, 25)), rel_tol=1e-9)
    assert math.isclose(d['iqr_75'], float(np.percentile(filtered_np, 75)), rel_tol=1e-9)

    d_empty = distribution_stats([None, None, None])
    assert d_empty['n_cases'] == 0
    for k in ('mean', 'std', 'median', 'iqr_25', 'iqr_75', 'min', 'max'):
        assert d_empty[k] is None
    assert not any(isinstance(d_empty[k], float) and math.isnan(d_empty[k])
                   for k in d_empty if k != 'n_cases')

    d_single = distribution_stats([0.42])
    assert d_single['n_cases'] == 1
    assert d_single['mean'] == 0.42 and d_single['std'] == 0.0

    print("distribution_stats: OK")

    # --- wilcoxon_paired ------------------------------------------------------
    a = [0.60, 0.62, 0.58, 0.65, 0.70, 0.55]
    b = [x + 0.1 for x in a]
    res = wilcoxon_paired(a, b)
    assert res['n_pairs'] == 6
    assert res['p_value'] is not None and res['p_value'] < 0.5
    assert res['effect_size_r'] is not None and res['effect_size_r'] != 0.0
    assert res['statistic'] is not None
    # b is uniformly larger than a -> a - b is uniformly negative -> sign < 0
    assert res['effect_size_r'] < 0

    res_identical = wilcoxon_paired(a, a)
    assert res_identical['statistic'] is None
    assert res_identical['p_value'] is None
    assert res_identical['effect_size_r'] is None
    assert res_identical['n_pairs'] == 6
    assert 'note' in res_identical and isinstance(res_identical['note'], str)

    # None-filtering: only cases present in both count as valid pairs.
    a_none = [0.5, None, 0.6, 0.7, None, 0.8]
    b_none = [0.4, 0.9, None, 0.6, 0.2, 0.7]
    res_none = wilcoxon_paired(a_none, b_none)
    # valid pairs: index0 (0.5,0.4), index3 (0.7,0.6), index5 (0.8,0.7) -> 3 pairs
    assert res_none['n_pairs'] == 3

    # Fewer than 2 valid pairs -> insufficient pairs, no raise.
    res_insufficient = wilcoxon_paired([0.5, None, None], [0.4, None, None])
    assert res_insufficient['n_pairs'] == 1
    assert res_insufficient['statistic'] is None
    assert res_insufficient['note'] == 'insufficient pairs'

    res_zero_pairs = wilcoxon_paired([None, None], [0.1, None])
    assert res_zero_pairs['n_pairs'] == 0
    assert res_zero_pairs['note'] == 'insufficient pairs'

    try:
        wilcoxon_paired([0.1, 0.2], [0.1, 0.2, 0.3])
        raise AssertionError("expected ValueError on mismatched lengths")
    except ValueError:
        pass

    print("wilcoxon_paired: OK")

    # --- fairness_block ---------------------------------------------------
    scores = {0: 0.9, 1: 0.5, 2: 0.7, 3: 0.6}
    modalities = {
        0: [True, False, False, True],   # FLAIR, T2
        1: [False, True, False, False],  # T1ce
        2: [True, True, False, False],   # FLAIR, T1ce
        3: [False, False, True, True],   # T1, T2
    }
    fb = fairness_block(scores, modalities)

    assert fb['worst_client'] == 1
    assert fb['best_client'] == 0
    assert fb['worst_client_score'] == 0.5
    assert fb['best_client_score'] == 0.9

    manual_scores = np.array([0.9, 0.5, 0.7, 0.6])
    assert math.isclose(fb['std_across_clients'], float(np.std(manual_scores, ddof=0)), rel_tol=1e-9)
    assert math.isclose(fb['range'], 0.4, rel_tol=1e-9)

    # 25th percentile of [0.9,0.5,0.7,0.6] is 0.575 -> only client 1 (0.5) qualifies.
    assert math.isclose(fb['worst_quartile_mean'], 0.5, rel_tol=1e-9)

    pm = fb['per_modality_subset_aggregates']
    assert pm['FLAIR']['n_clients'] == 2 and math.isclose(pm['FLAIR']['mean_score'], (0.9 + 0.7) / 2, rel_tol=1e-9)
    assert pm['T1ce']['n_clients'] == 2 and math.isclose(pm['T1ce']['mean_score'], (0.5 + 0.7) / 2, rel_tol=1e-9)
    assert pm['T1']['n_clients'] == 1 and math.isclose(pm['T1']['mean_score'], 0.6, rel_tol=1e-9)
    assert pm['T2']['n_clients'] == 2 and math.isclose(pm['T2']['mean_score'], (0.9 + 0.6) / 2, rel_tol=1e-9)

    # Modality nobody holds -> explicit None mean, not a crash / NaN.
    modalities_gap = {
        0: [True, False, False, False],
        1: [True, False, False, False],
    }
    scores_gap = {0: 0.8, 1: 0.6}
    fb_gap = fairness_block(scores_gap, modalities_gap)
    assert fb_gap['per_modality_subset_aggregates']['T1ce']['n_clients'] == 0
    assert fb_gap['per_modality_subset_aggregates']['T1ce']['mean_score'] is None

    # Single-client edge case: bottom-quartile fallback still yields the one client.
    fb_one = fairness_block({0: 0.77}, {0: [True, True, True, True]})
    assert fb_one['worst_client'] == 0 and fb_one['best_client'] == 0
    assert fb_one['worst_quartile_mean'] == 0.77
    assert fb_one['std_across_clients'] == 0.0

    print("fairness_block: OK")

    print("ALL TESTS PASSED")

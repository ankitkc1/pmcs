"""
Per-modality-encoder participation/staleness instrumentation for a
federated learning server, plus a closed-form prediction of how often a
modality-specific encoder goes a round without any contributor.

Setup: there are 4 modality-specific encoders (order [FLAIR, T1ce, T1, T2]).
Each round, some subset of the K participating clients hold each modality
(a "pool"); an encoder is FedAvg'd that round only among clients that (a)
participated this round AND (b) hold that modality. If zero such clients
exist that round, the encoder is not updated ("idle" / stale that round).

This module does not touch training, aggregation, or data loading -- it is
pure bookkeeping/prediction, meant to be fed contributor counts computed
elsewhere (e.g. by whatever selects participants and intersects with each
modality's client pool each round).
"""
import math
from random import Random

MODALITY_NAMES = ['FLAIR', 'T1ce', 'T1', 'T2']


class EncoderCoverageTracker:
    """Tracks, per round and per modality, whether that modality's encoder
    received any FedAvg contributors, how many, and how many consecutive
    rounds it has gone without one (staleness)."""

    def __init__(self, modality_names=MODALITY_NAMES):
        self.modality_names = list(modality_names)
        self._staleness = {m: 0 for m in self.modality_names}
        self._rounds = {}

    def record_round(self, round_idx, contributor_counts):
        """contributor_counts: length-len(modality_names) sequence of ints;
        contributor_counts[m] = number of participating-this-round clients
        that both participated AND hold modality m."""
        if len(contributor_counts) != len(self.modality_names):
            raise ValueError(
                "contributor_counts length %d does not match %d modalities"
                % (len(contributor_counts), len(self.modality_names))
            )

        round_record = {}
        for m, n_contributors in zip(self.modality_names, contributor_counts):
            was_updated = n_contributors > 0
            if was_updated:
                self._staleness[m] = 0
                staleness = 0
            else:
                self._staleness[m] += 1
                staleness = self._staleness[m]
            round_record[m] = {
                'was_updated': was_updated,
                'n_contributors': int(n_contributors),
                'staleness': staleness,
            }
        self._rounds[round_idx] = round_record

    def per_round(self):
        return self._rounds


def predicted_idle_fraction(N, n_m, K):
    """Probability that a uniformly-random sample of K clients out of N
    contains ZERO of the n_m clients holding modality m.

    Closed form (hypergeometric, "none of the successes drawn"):
        C(N - n_m, K) / C(N, K)

    math.comb(a, b) returns 0 whenever b > a >= 0 (Python's definition of
    "choose" for k > n), so the K > N - n_m case (drawing K clients while
    avoiding all n_m holders is impossible) falls out naturally as 0.0
    without a special case. N - n_m can never be negative since every
    modality's pool size n_m <= N by construction.
    """
    numerator = math.comb(N - n_m, K)
    denominator = math.comb(N, K)
    return numerator / denominator


def monte_carlo_idle_fraction(N, n_m, K, num_trials=20000, seed=0):
    """Empirical cross-check for predicted_idle_fraction: repeatedly sample
    K indices out of N (without replacement) and check whether none of them
    fall among a fixed set of n_m "holder" indices (0..n_m-1). Uses a local
    random.Random instance so results are reproducible and independent of
    any other code's use of the global `random` module or numpy RNG."""
    rng = Random(seed)
    population = range(N)
    holders = set(range(n_m))
    miss_count = 0
    for _ in range(num_trials):
        sample = rng.sample(population, K)
        if holders.isdisjoint(sample):
            miss_count += 1
    return miss_count / num_trials


def summarize(tracker, pool_sizes, total_clients, participation_k):
    """pool_sizes: {'FLAIR': n_m, ...} -- how many of total_clients hold
    that modality; fixed by the split design, independent of who
    participates in any given round. participation_k: how many clients
    are sampled/active per round (== total_clients for full participation).
    """
    per_round = tracker.per_round()
    modality_names = tracker.modality_names
    total_rounds = len(per_round)

    per_modality = {}
    for m in modality_names:
        updates = [per_round[r][m] for r in per_round]
        update_count = sum(1 for u in updates if u['was_updated'])
        idle_rounds = sum(1 for u in updates if not u['was_updated'])
        idle_fraction = idle_rounds / total_rounds if total_rounds > 0 else 0.0
        stalenesses = [u['staleness'] for u in updates]
        contributors = [u['n_contributors'] for u in updates]
        mean_staleness = sum(stalenesses) / len(stalenesses) if stalenesses else 0.0
        max_staleness = max(stalenesses) if stalenesses else 0
        # T * (1 - idle_fraction): how many of the T observed rounds this
        # modality's encoder actually received a real update in (a finite
        # number always, including 0.0 when total_rounds is 0) -- this is
        # numerically identical to update_count, exposed under this name for
        # report-schema consistency with the partial-participation spec.
        effective_horizon = total_rounds * (1.0 - idle_fraction)
        mean_contributors = sum(contributors) / len(contributors) if contributors else 0.0

        per_modality[m] = {
            'update_count': update_count,
            'idle_rounds': idle_rounds,
            'idle_fraction': idle_fraction,
            'mean_staleness': mean_staleness,
            'max_staleness': max_staleness,
            'effective_horizon': effective_horizon,
            'mean_contributors': mean_contributors,
            'pool_size': pool_sizes[m],
        }

    predicted_idle = {
        m: predicted_idle_fraction(total_clients, pool_sizes[m], participation_k)
        for m in modality_names
    }

    return {
        'per_modality': per_modality,
        'per_round': per_round,
        'predicted_idle_fraction': predicted_idle,
    }


if __name__ == '__main__':
    # --- Case 1: K == N == 8 (full participation), both splits' pools ----
    split_a_pools = {'FLAIR': 7, 'T1ce': 2, 'T1': 5, 'T2': 4}
    split_b_pools = {'FLAIR': 2, 'T1ce': 7, 'T1': 5, 'T2': 4}

    for pools in (split_a_pools, split_b_pools):
        for m in MODALITY_NAMES:
            n_m = pools[m]
            pred = predicted_idle_fraction(8, n_m, 8)
            assert pred == 0.0, (
                "expected exact 0.0 idle fraction at K==N==8 for %s (n_m=%d), got %r"
                % (m, n_m, pred)
            )

    # 150 synthetic rounds of true 8/8 participation: every round's
    # contributor_counts == the full pool (nobody ever drops out).
    for pools in (split_a_pools, split_b_pools):
        tracker = EncoderCoverageTracker()
        full_counts = [pools[m] for m in MODALITY_NAMES]
        for r in range(150):
            tracker.record_round(r, full_counts)

        summary = summarize(tracker, pools, total_clients=8, participation_k=8)
        assert len(summary['per_round']) == 150

        for m in MODALITY_NAMES:
            stats = summary['per_modality'][m]
            assert stats['idle_fraction'] == 0.0, (m, stats['idle_fraction'])
            assert stats['update_count'] == 150, (m, stats['update_count'])
            assert stats['idle_rounds'] == 0
            assert stats['mean_contributors'] == pools[m], (
                m, stats['mean_contributors'], pools[m]
            )
            assert stats['mean_staleness'] == 0.0
            assert stats['max_staleness'] == 0
            # effective_horizon = T * (1 - idle_fraction); at idle_fraction
            # == 0.0 over 150 observed rounds this is exactly 150.0 (every
            # round was a real update), and always equals update_count.
            assert stats['effective_horizon'] == 150.0, stats['effective_horizon']
            assert stats['effective_horizon'] == float(stats['update_count'])
            assert summary['predicted_idle_fraction'][m] == 0.0

    # --- Case 2: synthetic 5-round run at K=3, N=8, n_m=2 -----------------
    # Exercise the tracker's staleness bookkeeping with a hand-picked,
    # non-trivial contributor sequence for the modality under test (index 0
    # == FLAIR by convention here), while the other 3 modalities are always
    # covered so only FLAIR's staleness trajectory is being asserted.
    tracker2 = EncoderCoverageTracker()
    flair_contributors_by_round = [0, 0, 1, 0, 2]  # idle, idle, hit, idle, hit
    other_contributors = [3, 3, 3]  # always covered, irrelevant to the assertions
    for r, fc in enumerate(flair_contributors_by_round):
        tracker2.record_round(r, [fc] + other_contributors)

    per_round2 = tracker2.per_round()
    expected_staleness = [1, 2, 0, 1, 0]
    expected_was_updated = [False, False, True, False, True]
    for r in range(5):
        rec = per_round2[r]['FLAIR']
        assert rec['staleness'] == expected_staleness[r], (r, rec)
        assert rec['was_updated'] == expected_was_updated[r], (r, rec)
        assert rec['n_contributors'] == flair_contributors_by_round[r]

    # effective_horizon = T * (1 - idle_fraction) on this same 5-round run:
    # FLAIR was idle in rounds 0,1,3 (3 idle rounds) and updated in 2,4 (2
    # updates) out of 5 -- idle_fraction = 3/5 = 0.6, so
    # effective_horizon = 5 * (1 - 0.6) = 2.0, exactly equal to update_count.
    summary2 = summarize(tracker2, pool_sizes={'FLAIR': 2, 'T1ce': 3, 'T1': 3, 'T2': 3},
                          total_clients=8, participation_k=3)
    flair_stats = summary2['per_modality']['FLAIR']
    assert flair_stats['update_count'] == 2
    assert flair_stats['idle_rounds'] == 3
    assert abs(flair_stats['idle_fraction'] - 0.6) < 1e-12
    assert abs(flair_stats['effective_horizon'] - 2.0) < 1e-12
    assert flair_stats['effective_horizon'] == float(flair_stats['update_count'])
    print('summarize(): effective_horizon = T*(1-idle_fraction) matches update_count under partial idle: OK')

    pred = predicted_idle_fraction(8, 2, 3)
    mc = monte_carlo_idle_fraction(8, 2, 3, num_trials=50000, seed=0)
    assert abs(pred - mc) <= 0.02, (
        "closed-form (%r) and Monte Carlo (%r) predicted idle fractions disagree by more than 0.02"
        % (pred, mc)
    )
    # sanity: exact value via direct comb arithmetic
    assert pred == math.comb(6, 3) / math.comb(8, 3)

    # --- Case 3: manual sanity case, exact fraction -----------------------
    # N=10, n_m=1, K=5 -> C(9,5)/C(10,5) = 126/252 = 0.5 exactly.
    exact_case = predicted_idle_fraction(10, 1, 5)
    assert math.comb(9, 5) == 126
    assert math.comb(10, 5) == 252
    assert exact_case == 126 / 252 == 0.5, exact_case

    # --- Edge case: K > N - n_m (impossible to avoid every holder) -------
    # e.g. N=8, n_m=2, K=7: N-n_m=6 < K=7, so math.comb(6,7) must be 0,
    # giving predicted_idle_fraction == 0.0 with no special-casing needed.
    assert math.comb(6, 7) == 0
    assert predicted_idle_fraction(8, 2, 7) == 0.0

    # --- Edge case: K >= N (sampling everyone or more) --------------------
    # Every modality has n_m >= 1 by construction, so missing it is
    # impossible once K >= N.
    for n_m in (1, 2, 5, 7, 8):
        assert predicted_idle_fraction(8, n_m, 8) == 0.0

    # Sanity check on the known ground-truth encoder/shared-base constants
    # (this module does not itself implement communication-byte accounting
    # -- that lives elsewhere -- but the constants it was specified against
    # should still hold together): a hypothetical client holding all 4
    # modalities would upload 4*1,466,496 + 2,540,732 = 8,406,716 params.
    assert 4 * 1_466_496 + 2_540_732 == 8_406_716

    print("ALL TESTS PASSED")

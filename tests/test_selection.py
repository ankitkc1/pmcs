
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import train_federated as tf
from utils.selection import ClientSelector

# Scenario 1 manifest (Split A), columns FLAIR / T1ce / T1 / T2. Identical to
# dataset/split_io.py's expected_split_a_masks and configs/splitA.json.
SCENARIO_1_MANIFEST = np.array([
    [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0],
    [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1],
])


class _PoisonCtx:
    """ctx whose __getattr__ raises on any access -- see test 3."""

    def __getattr__(self, name):
        raise AssertionError('ctx.{} was accessed'.format(name))


class _FakeCtx:
    """Minimal working ctx for poc: n_samples plus a deterministic-but-varying
    local_loss, with no model/data involved."""

    def __init__(self, n_samples, seed=0):
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    def local_loss(self, client_ids):
        return {k: float(self._rng.random()) for k in client_ids}


def _aggregated_modalities(manifest, selected):
    return [any(manifest[k, m] for k in selected) for m in range(manifest.shape[1])]


# ---------------------------------------------------------------------------
# Test 1: reproduction -- uniform must exactly reproduce the pre-refactor code
# ---------------------------------------------------------------------------

def test_1_reproduction():
    seed, K, N, T = 42, 2, 8, 150
    manifest = SCENARIO_1_MANIFEST

    selector = ClientSelector('uniform', manifest, K, seed)
    old_state = {'client_num': N, 'clients_per_round': K, 'participation_rng': random.Random(seed)}

    for t in range(T):
        new_selected = selector.select(t, ctx=None)
        old_selected = tf.select_clients(t, old_state)
        assert new_selected == old_selected, (
            'first divergence at round {}: new={} old={}'.format(t, new_selected, old_selected))
        selector.observe(t, new_selected, [True] * manifest.shape[1])

    print('1 (reproduction: uniform seed=42 K=2 T=150 matches train_federated.select_clients exactly): PASS')


# ---------------------------------------------------------------------------
# Test 2: simulator parity -- the training-loop mics must match the CPU
# simulator's validated numbers (147 worst-encoder rounds, max stall 1)
# ---------------------------------------------------------------------------

def test_2_simulator_parity():
    K, T, beta, s_max = 2, 150, 0.15, 12
    manifest = SCENARIO_1_MANIFEST
    M = manifest.shape[1]

    selector = ClientSelector('mics', manifest, K, seed=42, beta=beta, s_max=s_max)

    for t in range(T):
        selected = selector.select(t, ctx=None)
        selector.observe(t, selected, _aggregated_modalities(manifest, selected))

    worst_encoder_coverage = min(selector.updates)
    assert worst_encoder_coverage >= 145, (
        'worst-encoder coverage {} < 145 -- the training-loop mics is not the algorithm the CPU '
        'simulator validated (scripts/mics_simulator.py gives 147)'.format(worst_encoder_coverage))

    history = selector.stats()['history']
    max_stall = 0
    for m in range(M):
        streak = 0
        for record in history:
            if record['aggregated_modalities'][m]:
                streak = 0
            else:
                streak += 1
                max_stall = max(max_stall, streak)
    assert max_stall <= 2, (
        'max consecutive encoder stall {} > 2 -- the training-loop mics is not the algorithm the CPU '
        'simulator validated (scripts/mics_simulator.py gives 1)'.format(max_stall))

    print('2 (simulator parity: worst-encoder coverage={} (>=145), max consecutive stall={} (<=2)): PASS'.format(
        worst_encoder_coverage, max_stall))


# ---------------------------------------------------------------------------
# Test 3: mics needs nothing from ctx; poc does
# ---------------------------------------------------------------------------

def test_3_mics_needs_nothing():
    K, T = 2, 150
    manifest = SCENARIO_1_MANIFEST
    poison = _PoisonCtx()

    mics_selector = ClientSelector('mics', manifest, K, seed=7, beta=0.15, s_max=12)
    for t in range(T):
        selected = mics_selector.select(t, poison)
        mics_selector.observe(t, selected, _aggregated_modalities(manifest, selected))
    print('3a (mics completes {} rounds against a ctx that raises on any attribute access): PASS'.format(T))

    poc_selector = ClientSelector('poc', manifest, K, seed=7, d=2 * K)
    # Round 0 is the shared uniform fallback for every policy -- must NOT touch ctx.
    selected0 = poc_selector.select(0, poison)
    poc_selector.observe(0, selected0, _aggregated_modalities(manifest, selected0))

    try:
        poc_selector.select(1, poison)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            'poc.select() at t=1 did not touch ctx -- it should have needed n_samples/local_loss')

    print('3b (poc raises on the poisoned ctx at t=1, once it needs a real candidate evaluation): PASS')


# ---------------------------------------------------------------------------
# Test 4: well-formedness, every policy; forced inclusion specifically for mics
# ---------------------------------------------------------------------------

def test_4_well_formedness():
    manifest = SCENARIO_1_MANIFEST
    N, M = manifest.shape
    K = 2
    T = 60
    n_samples = {k: 10 * (k + 1) for k in range(N)}

    for policy in ClientSelector.POLICIES:
        kwargs = {}
        s_max = 12
        if policy == 'poc':
            kwargs['d'] = 2 * K
        if policy == 'mics':
            s_max = 3  # small on purpose: exercise forced inclusion within T=60 rounds
            kwargs['beta'] = 0.15
            kwargs['s_max'] = s_max

        selector = ClientSelector(policy, manifest, K, seed=3, **kwargs)
        ctx = _FakeCtx(n_samples, seed=3) if policy == 'poc' else None
        forced_events = 0

        for t in range(T):
            pre_stale = list(selector.stale)
            selected = selector.select(t, ctx)

            assert len(selected) == K, (policy, t, selected)
            assert len(set(selected)) == K, (policy, t, selected)
            assert all(0 <= k < N for k in selected), (policy, t, selected)
            assert selected == sorted(selected), (policy, t, selected)

            if policy == 'mics':
                qualifying = [k for k in range(N) if pre_stale[k] >= s_max]
                if qualifying:
                    forced_events += 1
                    if len(qualifying) <= K:
                        for k in qualifying:
                            assert k in selected, (
                                'client {} has stale={} >= s_max={} but was not selected at round {} '
                                '(only {} clients qualified, <= K={})'.format(
                                    k, pre_stale[k], s_max, t, len(qualifying), K))

            selector.observe(t, selected, _aggregated_modalities(manifest, selected))

        if policy == 'mics':
            assert forced_events > 0, (
                'forced inclusion never fired at s_max={} over {} rounds -- test is vacuous'.format(s_max, T))
            print('4 ({}, well-formedness incl. forced-inclusion, s_max={}, fired in {}/{} rounds): PASS'.format(
                policy, s_max, forced_events, T))
        else:
            print('4 ({}, well-formedness over {} rounds): PASS'.format(policy, T))


# ---------------------------------------------------------------------------
# Test 5: refutation -- all-ones manifest must reduce mics to near-round-robin
# ---------------------------------------------------------------------------

def test_5_refutation_all_ones_manifest():
    N, M, K, T = 8, 4, 2, 150
    manifest = np.ones((N, M), dtype=np.int64)
    seed = 11

    mics_selector = ClientSelector('mics', manifest, K, seed, beta=0.15, s_max=12)
    participation = [0] * N
    for t in range(T):
        selected = mics_selector.select(t, ctx=None)
        for k in selected:
            participation[k] += 1
        mics_selector.observe(t, selected, _aggregated_modalities(manifest, selected))

    values = np.asarray(participation, dtype=float)
    jain = float(values.sum() ** 2 / (N * np.sum(values ** 2)))
    assert jain > 0.99, (
        'client participation Jain={:.4f} <= 0.99 on an all-ones manifest -- theory predicts near-round-robin '
        '(cover(S) is identical for every size-K S here, so selection should be driven entirely by the '
        'fairness term)'.format(jain))

    mics_worst_coverage = min(mics_selector.updates)

    uniform_selector = ClientSelector('uniform', manifest, K, seed)
    for t in range(T):
        selected = uniform_selector.select(t, ctx=None)
        uniform_selector.observe(t, selected, _aggregated_modalities(manifest, selected))
    uniform_worst_coverage = min(uniform_selector.updates)

    assert mics_worst_coverage == T and uniform_worst_coverage == T, (
        'expected perfect (T={}) encoder coverage for both policies on an all-ones manifest, got '
        'mics={} uniform={}'.format(T, mics_worst_coverage, uniform_worst_coverage))
    assert mics_worst_coverage == uniform_worst_coverage, (
        'mics={} vs uniform={} encoder coverage on an all-ones manifest -- theory predicts an exact tie; '
        'a difference means the mechanism is not what is claimed'.format(
            mics_worst_coverage, uniform_worst_coverage))

    print('5 (refutation: all-ones manifest -> client participation Jain={:.4f} (>0.99), '
          'encoder coverage tie at {}/{}): PASS'.format(jain, mics_worst_coverage, T))


if __name__ == '__main__':
    test_1_reproduction()
    test_2_simulator_parity()
    test_3_mics_needs_nothing()
    test_4_well_formedness()
    test_5_refutation_all_ones_manifest()
    print()
    print('ALL SELECTION TESTS PASSED (1-5)')

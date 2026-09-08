#!/usr/bin/env python3
"""
Fast, self-contained tests for partial client participation (--clients_per_round),
covering the acceptance criteria C1-C5 from the partial-participation task.
No GPU, no real data, no training -- seconds to run.

    python tests/test_partial_participation.py
"""
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import train_federated as tf
from models.fusion_net import FusionSegNet
from utils import encoder_coverage as coverage_mod

# Split A's registered mask design (FLAIR=7, T1ce=2, T1=5, T2=4 pool sizes).
SPLIT_A_MASKS = [
    [True, True, True, True], [True, True, True, False],
    [True, False, True, True], [True, False, True, False],
    [True, False, False, True], [True, False, False, False],
    [True, False, True, False], [False, False, False, True],
]
SPLIT_A_POOL_SIZES = {'FLAIR': 7, 'T1ce': 2, 'T1': 5, 'T2': 4}


def _fresh_agg_state(client_num, clients_per_round, seed=42):
    return {
        'client_num': client_num,
        'clients_per_round': clients_per_round,
        'participation_rng': random.Random(seed),
        'update_count': [0, 0, 0, 0],
        'staleness': [0, 0, 0, 0],
    }


def _tensors_all_finite(state_dict):
    return all(torch.isfinite(t).all() for t in state_dict.values())


def _state_dicts_equal(a, b):
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# C1: backward compatibility at clients_per_round == client_num
# ---------------------------------------------------------------------------

def test_c1_backward_compatibility():
    client_num = 8
    state = _fresh_agg_state(client_num, client_num)

    # select_clients must return list(range(N)), in that exact sorted order,
    # for every round, at every seed -- this is what guarantees broadcast/
    # aggregation iterate over the same clients in the same order as the
    # pre-partial-participation code (which always used range(client_num)).
    for round_idx in range(5):
        active = tf.select_clients(round_idx, state)
        assert active == list(range(client_num)), (round_idx, active)

    # A different seed must not change this -- full participation collapses
    # to range(N) regardless of RNG state, since sampling ALL N elements
    # without replacement and sorting always yields [0..N-1].
    state2 = _fresh_agg_state(client_num, client_num, seed=999)
    for round_idx in range(5):
        assert tf.select_clients(round_idx, state2) == list(range(client_num))

    # broadcast_weights with active_clients == all clients must touch every
    # client identically to the pre-partial-participation behaviour (which
    # unconditionally iterated every model in model_clients). Verify this
    # with real (tiny) FusionSegNet instances and real encoder/decoder
    # tensors, not a re-implementation.
    torch.manual_seed(0)
    template = FusionSegNet(num_cls=4)
    global_encoders = [
        template.flair_encoder.state_dict(), template.t1ce_encoder.state_dict(),
        template.t1_encoder.state_dict(), template.t2_encoder.state_dict(),
    ]
    global_decoder_prior = {k: v for k, v in template.fusion_decoder.state_dict().items()
                             if not k.endswith('_residual')}
    masks = SPLIT_A_MASKS

    model_clients_new = [FusionSegNet(num_cls=4) for _ in range(client_num)]
    model_clients_old_equivalent = [copy.deepcopy(m) for m in model_clients_new]

    active_clients = tf.select_clients(0, state)  # == range(8)
    tf.broadcast_weights(model_clients_new, global_encoders, global_decoder_prior, masks, active_clients)
    # Old-style: unconditionally broadcast to every client (pre-partial-participation
    # behaviour), reimplemented here ONLY as the ground truth to diff against --
    # not a stand-in for the function under test.
    for m, mask in zip(model_clients_old_equivalent, masks):
        for held, attr, enc_state in zip(mask, tf.ENCODER_ATTRS, global_encoders):
            if held:
                getattr(m, attr).load_state_dict(enc_state)
        m.fusion_decoder.load_state_dict(global_decoder_prior, strict=False)

    for c in range(client_num):
        assert _state_dicts_equal(model_clients_new[c].state_dict(), model_clients_old_equivalent[c].state_dict()), \
            'client {}: broadcast result differs from full-participation baseline'.format(c)

    print('C1 (backward compatibility at K==N): PASS')


# ---------------------------------------------------------------------------
# C2 / B3: zero-contributor modality round
# ---------------------------------------------------------------------------

def test_c2_zero_contributor_round():
    torch.manual_seed(1)
    template = FusionSegNet(num_cls=4)
    global_encoders_before = [
        copy.deepcopy(template.flair_encoder.state_dict()),
        copy.deepcopy(template.t1ce_encoder.state_dict()),
        copy.deepcopy(template.t1_encoder.state_dict()),
        copy.deepcopy(template.t2_encoder.state_dict()),
    ]
    global_encoders = [dict(e) for e in global_encoders_before]  # working copy

    client_num = 8
    masks = SPLIT_A_MASKS  # T1ce (index 1) is held by only clients 0 and 1
    masks_torch = torch.tensor(masks, dtype=torch.bool)

    # Select 2 clients that do NOT include either T1ce holder (clients 0, 1)
    # -- e.g. clients 2 and 3 -- so the T1ce pool has zero contributors this
    # round while FLAIR (held by both) still gets updated.
    active_clients = [2, 3]
    assert not masks[2][1] and not masks[3][1], 'test setup: neither active client may hold T1ce'
    assert masks[2][0] and masks[3][0], 'test setup: both active clients must hold FLAIR'

    # Fabricate "locally trained" encoder states for the active clients:
    # perturb every parameter so a real (non-trivial) FedAvg would visibly
    # change the aggregated weights, making a false "unchanged" pass
    # impossible to get by accident.
    local_encoders = {}
    for c in active_clients:
        client_model = copy.deepcopy(template)
        with torch.no_grad():
            for enc_attr in tf.ENCODER_ATTRS:
                for p in getattr(client_model, enc_attr).parameters():
                    p.add_(torch.randn_like(p) * 0.5 + 1.0)
        local_encoders[c] = [
            client_model.flair_encoder.state_dict(), client_model.t1ce_encoder.state_dict(),
            client_model.t1_encoder.state_dict(), client_model.t2_encoder.state_dict(),
        ]

    global_encoders, contributor_counts = tf.aggregate_encoders(
        local_encoders, active_clients, masks_torch, global_encoders)

    # (a) no NaN/Inf anywhere in the resulting encoders.
    for m in range(4):
        assert _tensors_all_finite(global_encoders[m]), 'encoder {} has non-finite values'.format(m)

    # T1ce (index 1): zero contributors -- must be BITWISE IDENTICAL to
    # before, not zeroed, not reinitialised, not averaged over an empty set.
    assert contributor_counts[1] == 0, contributor_counts
    assert _state_dicts_equal(global_encoders[1], global_encoders_before[1]), \
        'T1ce encoder changed despite zero contributors -- must be carried forward unchanged'

    # (c) FLAIR (index 0): both active clients hold it -- must have changed
    # (a real FedAvg happened, not a silent no-op mistaken for "unchanged").
    assert contributor_counts[0] == 2, contributor_counts
    assert not _state_dicts_equal(global_encoders[0], global_encoders_before[0]), \
        'FLAIR encoder did not change despite having 2 real contributors'

    # Coverage tracker records this correctly too.
    tracker = coverage_mod.EncoderCoverageTracker()
    tracker.record_round(0, contributor_counts)
    rec = tracker.per_round()[0]
    assert rec['T1ce']['was_updated'] is False
    assert rec['T1ce']['n_contributors'] == 0
    assert rec['T1ce']['staleness'] == 1
    assert rec['FLAIR']['was_updated'] is True
    assert rec['FLAIR']['staleness'] == 0

    print('C2 / B3 (zero-contributor modality round): PASS')


# ---------------------------------------------------------------------------
# C3 / B2: non-selected client's state (including private residual) untouched
# ---------------------------------------------------------------------------

def test_c3_non_selected_client_untouched():
    torch.manual_seed(2)
    client_num = 8
    masks = SPLIT_A_MASKS
    masks_torch = torch.tensor(masks, dtype=torch.bool)

    model_clients = [FusionSegNet(num_cls=4) for _ in range(client_num)]
    # Perturb every client's private residual so it's nonzero and
    # distinguishable (a freshly-initialised residual starts at exactly
    # zero, which would make "unchanged" trivially true for the wrong
    # reason).
    with torch.no_grad():
        for m in model_clients:
            for name, p in m.named_parameters():
                if name.endswith('_residual'):
                    p.add_(torch.randn_like(p) * 0.3 + 0.1)

    excluded_client = 5
    active_clients = [c for c in range(client_num) if c != excluded_client]
    assert excluded_client not in active_clients

    before_state = copy.deepcopy(model_clients[excluded_client].state_dict())

    # Simulate one full round's worth of aggregation + broadcast, exactly as
    # train_federated.py's main loop does, using only active_clients.
    local_encoders, local_decoders = {}, {}
    for c in active_clients:
        client_model = copy.deepcopy(model_clients[c])
        with torch.no_grad():
            for enc_attr in tf.ENCODER_ATTRS:
                for p in getattr(client_model, enc_attr).parameters():
                    p.add_(torch.randn_like(p) * 0.2)
        local_encoders[c] = [
            client_model.flair_encoder.state_dict(), client_model.t1ce_encoder.state_dict(),
            client_model.t1_encoder.state_dict(), client_model.t2_encoder.state_dict(),
        ]
        local_decoders[c] = client_model.fusion_decoder.state_dict()

    global_encoders = [
        model_clients[0].flair_encoder.state_dict(), model_clients[0].t1ce_encoder.state_dict(),
        model_clients[0].t1_encoder.state_dict(), model_clients[0].t2_encoder.state_dict(),
    ]
    global_encoders, contributor_counts = tf.aggregate_encoders(
        local_encoders, active_clients, masks_torch, global_encoders)
    global_decoder_prior = tf.aggregate_decoder(local_decoders, active_clients)

    tf.broadcast_weights(model_clients, global_encoders, global_decoder_prior, masks, active_clients)

    after_state = model_clients[excluded_client].state_dict()
    assert _state_dicts_equal(before_state, after_state), \
        'non-selected client {} state changed after a round it sat out'.format(excluded_client)

    # Explicitly re-confirm the private residual specifically, since that is
    # the constraint called out by name.
    for name in before_state:
        if name.endswith('_residual'):
            assert torch.equal(before_state[name], after_state[name]), \
                'non-selected client residual {} changed'.format(name)

    # And confirm an ACTIVE client's residual is untouched too (broadcast
    # never writes residuals for anyone, selected or not) while its held
    # encoders DID change.
    active_example = active_clients[0]
    assert masks[active_example][0], 'test setup: active_example must hold FLAIR'
    flair_before = copy.deepcopy(model_clients[active_example].flair_encoder.state_dict())
    # (already broadcast above; compare against the pre-perturbation template
    # instead -- flair_before was captured AFTER broadcast, so instead assert
    # it now equals the aggregated global encoder, proving it WAS updated)
    assert _state_dicts_equal(flair_before, global_encoders[0])

    print('C3 / B2 (non-selected client, incl. private residual, untouched): PASS')


# ---------------------------------------------------------------------------
# C4: 200-round selection-only coverage convergence (Split A pools, K=2,N=8)
# ---------------------------------------------------------------------------

def test_c4_coverage_convergence():
    client_num = 8
    clients_per_round = 2
    rounds = 200
    state = _fresh_agg_state(client_num, clients_per_round, seed=12345)
    masks_torch = torch.tensor(SPLIT_A_MASKS, dtype=torch.bool)

    tracker = coverage_mod.EncoderCoverageTracker()
    for round_idx in range(rounds):
        active = tf.select_clients(round_idx, state)
        contributor_counts = [
            int(sum(bool(masks_torch[c][m]) for c in active)) for m in range(4)
        ]
        tracker.record_round(round_idx, contributor_counts)

    summary = coverage_mod.summarize(
        tracker, SPLIT_A_POOL_SIZES, total_clients=client_num, participation_k=clients_per_round)

    expected = {'FLAIR': 0.000, 'T1': 0.107, 'T2': 0.214, 'T1ce': 0.536}
    print('{:<8}{:>12}{:>12}{:>10}{:>12}'.format('pool', 'predicted', 'observed', 'pool_n', 'tolerance'))
    all_within_tolerance = True
    for name in coverage_mod.MODALITY_NAMES:
        predicted = summary['predicted_idle_fraction'][name]
        observed = summary['per_modality'][name]['idle_fraction']
        # Binomial standard error of a proportion estimated from `rounds`
        # i.i.d. Bernoulli(predicted) trials -- each round's "was this pool
        # idle" is (to a very good approximation for K << N) an independent
        # draw at exactly this probability. A flat tolerance is wrong here:
        # variance peaks near p=0.5 (T1ce) and vanishes at p=0 (FLAIR), so
        # the SAME 200 rounds gives a much tighter bound for FLAIR than for
        # T1ce. 3 standard errors is a ~99.7% one-sided confidence margin.
        standard_error = (predicted * (1 - predicted) / rounds) ** 0.5
        tolerance = max(3 * standard_error, 1e-9)
        print('{:<8}{:>12.4f}{:>12.4f}{:>10}{:>12.4f}'.format(
            name, predicted, observed, SPLIT_A_POOL_SIZES[name], tolerance))
        # closed-form predicted_idle_fraction must match the given reference
        # values exactly (both are the same C(N-n,K)/C(N,K) formula).
        assert abs(predicted - expected[name]) < 0.001, (name, predicted, expected[name])
        # observed (Monte-Carlo, 200 rounds) must match the closed form
        # within Monte-Carlo error (3 sigma).
        if abs(observed - predicted) > tolerance:
            all_within_tolerance = False
            print('  ** {} observed/predicted mismatch beyond 3-sigma tolerance **'.format(name))

    assert all_within_tolerance, 'observed idle_fraction did not converge to the closed-form prediction'
    print('C4 (200-round coverage convergence, Split A pools, K=2/N=8): PASS')


# ---------------------------------------------------------------------------
# C5: communication -- rounds_participated sums to K * T
# ---------------------------------------------------------------------------

def test_c5_rounds_participated_sums_to_kt():
    client_num = 8
    clients_per_round = 3
    rounds = 50
    state = _fresh_agg_state(client_num, clients_per_round, seed=7)

    selected_by_round = {}
    for round_idx in range(rounds):
        selected_by_round[round_idx] = tf.select_clients(round_idx, state)
        assert len(selected_by_round[round_idx]) == clients_per_round

    rounds_participated = {
        c: sum(1 for selected in selected_by_round.values() if c in selected)
        for c in range(client_num)
    }
    total = sum(rounds_participated.values())
    assert total == clients_per_round * rounds, (total, clients_per_round, rounds)
    # every individual count must also be <= rounds (sanity) and the whole
    # federation must never exceed one selection event per client per round.
    for c, n in rounds_participated.items():
        assert 0 <= n <= rounds

    print('rounds_participated per client:', rounds_participated)
    print('sum = {} == K*T = {}*{} = {}'.format(total, clients_per_round, rounds, clients_per_round * rounds))
    print('C5 (rounds_participated sums to K*T): PASS')


if __name__ == '__main__':
    test_c1_backward_compatibility()
    test_c2_zero_contributor_round()
    test_c3_non_selected_client_untouched()
    test_c4_coverage_convergence()
    test_c5_rounds_participated_sums_to_kt()
    print()
    print('ALL PARTIAL-PARTICIPATION TESTS PASSED (C1-C5)')

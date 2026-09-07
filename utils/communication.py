"""
Participation-aware federated communication accounting.

Replaces a historical defect where every client's download was counted as
the full 4-modality model regardless of which modalities that client's mask
actually holds. The correct semantics, applied identically to upload and
download, is "you transmit/receive only what your mask holds": the modality
encoders in the client's own mask, plus the shared base (fusion prior +
shared decoder parts). The private per-client residual adapter is never
counted in upload or download -- it never leaves the client and the server
never has a copy to send back.

This module only does accounting (arithmetic over already-known parameter
counts and round/participation bookkeeping); it does not touch any model,
tensor, or training/aggregation code.
"""

ENCODER_PARAMS = 1_466_496
SHARED_BASE_PARAMS = 2_540_732
DEFAULT_DTYPE_BYTES = 4


def client_param_count(mask, encoder_params=ENCODER_PARAMS, shared_base=SHARED_BASE_PARAMS):
    """Parameter count for one direction (upload OR download -- "download
    what you need" makes the two formulas identical) for a client holding
    `mask` (length-4 bool/0/1, conventionally [FLAIR,T1ce,T1,T2], though this
    function only sums truthy entries so it is order-agnostic).
    """
    held = sum(1 for m in mask if m)
    return held * encoder_params + shared_base


class CommTracker:
    """Per-round, per-client upload/download bookkeeping.

    Only clients present in `active_clients` for a given round are recorded
    for that round at all -- an inactive client's per-round record simply has
    no entry for that round_idx, so a future partial-participation sweep
    never has to distinguish "recorded 0" from "did not participate".
    """

    def __init__(self, client_num, dtype_bytes=DEFAULT_DTYPE_BYTES):
        self.client_num = client_num
        self.dtype_bytes = dtype_bytes
        self.uploaded_params = {c: {} for c in range(client_num)}
        self.downloaded_params = {c: {} for c in range(client_num)}

    def record_round(self, round_idx, active_clients, masks):
        """active_clients: list of client indices selected this round.
        masks: dict or list mapping client_idx -> length-4 mask. Clients not
        in active_clients are not touched -- no zero entry, no entry at all.
        """
        for c in active_clients:
            n = client_param_count(masks[c])
            self.uploaded_params[c][round_idx] = n
            self.downloaded_params[c][round_idx] = n

    def summary(self):
        per_client = {}
        per_round = {}
        grand_uploaded = 0
        grand_downloaded = 0
        for c in range(self.client_num):
            up_rounds = self.uploaded_params[c]
            down_rounds = self.downloaded_params[c]
            up_total = sum(up_rounds.values())
            down_total = sum(down_rounds.values())
            grand_uploaded += up_total
            grand_downloaded += down_total
            per_client[c] = {
                'uploaded_params_total': up_total,
                'downloaded_params_total': down_total,
                'uploaded_bytes_total': up_total * self.dtype_bytes,
                'downloaded_bytes_total': down_total * self.dtype_bytes,
                'rounds_participated': len(up_rounds),
            }
            for round_idx, up_n in up_rounds.items():
                per_round.setdefault(round_idx, {})[c] = {
                    'up': up_n,
                    'down': down_rounds[round_idx],
                }
        total_bytes = (grand_uploaded + grand_downloaded) * self.dtype_bytes
        return {
            'dtype_bytes': self.dtype_bytes,
            'per_client': per_client,
            'per_round': per_round,
            'federation_totals': {
                'uploaded_params': grand_uploaded,
                'downloaded_params': grand_downloaded,
                'total_bytes': total_bytes,
            },
        }

    def state_dict(self):
        return {
            'uploaded': {c: dict(v) for c, v in self.uploaded_params.items()},
            'downloaded': {c: dict(v) for c, v in self.downloaded_params.items()},
        }

    def load_state_dict(self, state):
        self.uploaded_params = {
            int(c): {int(r): int(n) for r, n in v.items()} for c, v in state['uploaded'].items()
        }
        self.downloaded_params = {
            int(c): {int(r): int(n) for r, n in v.items()} for c, v in state['downloaded'].items()
        }


def bytes_to_target_dice(cumulative_bytes_by_round, mean_dice_pct_by_round, targets):
    """For each target dice percentage, the cumulative federation byte cost
    (upload+download so far) at the first round that reached it, scanning
    rounds in ascending order. None if a target is never reached.
    """
    rounds_sorted = sorted(mean_dice_pct_by_round.keys())
    result = {}
    for target in targets:
        hit_bytes = None
        for r in rounds_sorted:
            if mean_dice_pct_by_round[r] >= target:
                hit_bytes = cumulative_bytes_by_round.get(r)
                break
        result[str(target)] = hit_bytes
    return result


def parameter_breakdown(pool_sizes, client_num, adapter_residual_numel,
                         encoder_params=ENCODER_PARAMS, shared_base=SHARED_BASE_PARAMS):
    """Whole-federation parameter inventory.

    `pool_sizes` (e.g. {'FLAIR':7,'T1ce':2,'T1':5,'T2':4}) only sanity-checks
    that no modality's client count exceeds client_num; it does not enter the
    'total' formula. 'total' is a design choice, not a physical tensor: one
    copy each of the shared base and of all 4 encoder TYPES (the federation
    only ever needs to keep one canonical copy of each encoder type on the
    server, regardless of how many clients' masks include it), plus every
    client's own distinct private residual (those are never shared, so each
    client's copy really is a distinct set of parameters that exists
    somewhere in the federation). It intentionally does NOT multiply encoders
    by how many clients hold them -- that would double count the same server-
    side encoder weights once per holder.
    """
    assert all(0 <= v <= client_num for v in pool_sizes.values()), \
        'a modality pool size cannot exceed the number of clients'
    total = shared_base + 4 * encoder_params + client_num * adapter_residual_numel
    return {
        'total': total,
        'shared_base': shared_base,
        'per_encoder': encoder_params,
        'private_residual_per_client': adapter_residual_numel,
    }


def compression_ratio_vs_full_broadcast(masks, encoder_params=ENCODER_PARAMS, shared_base=SHARED_BASE_PARAMS):
    """Mean per-client transfer size (one direction) as a fraction of what a
    naive "broadcast the full 4-modality model to everyone" policy would
    cost -- the architectural bandwidth saving from modality-aware transfer.
    """
    counts = [client_param_count(mask) for mask in masks.values()]
    mean_count = sum(counts) / len(counts)
    full = client_param_count([True, True, True, True], encoder_params, shared_base)
    return mean_count / full


if __name__ == '__main__':
    def _masks_from_pool_sizes(pool_sizes_ordered, client_num):
        """Build an 8x4 (or NxM) boolean mask assignment realising exactly
        the given column sums (pool_sizes_ordered: list of 4 counts in
        [FLAIR,T1ce,T1,T2] order), by giving each modality to a distinct,
        cyclically-offset block of `count` clients. Offsets differ per
        modality so no single client is forced to hold all or none of them."""
        masks = {c: [False, False, False, False] for c in range(client_num)}
        for col, count in enumerate(pool_sizes_ordered):
            assert 0 <= count <= client_num
            offset = col * 2
            for i in range(count):
                c = (i + offset) % client_num
                masks[c][col] = True
        return masks

    # --- client_param_count -------------------------------------------------
    assert client_param_count([False, False, False, False]) == SHARED_BASE_PARAMS
    assert client_param_count([True, True, True, True]) == 4 * ENCODER_PARAMS + SHARED_BASE_PARAMS
    assert client_param_count([True, False, False, False]) == ENCODER_PARAMS + SHARED_BASE_PARAMS
    assert client_param_count([1, 0, 1, 0]) == 2 * ENCODER_PARAMS + SHARED_BASE_PARAMS  # int mask
    assert client_param_count((True, True, False, False)) == 2 * ENCODER_PARAMS + SHARED_BASE_PARAMS  # tuple, order-agnostic

    # --- CommTracker: the critical exact-equality gate ----------------------
    SPLIT_A_POOLS = [7, 2, 5, 4]   # FLAIR, T1ce, T1, T2
    SPLIT_B_POOLS = [2, 7, 5, 4]
    CLIENT_NUM = 8
    ROUNDS = 150

    for split_name, pools in (('Split A', SPLIT_A_POOLS), ('Split B', SPLIT_B_POOLS)):
        masks = _masks_from_pool_sizes(pools, CLIENT_NUM)
        # Verify the constructed masks really realise the requested pools.
        for col, expected_count in enumerate(pools):
            actual = sum(1 for c in range(CLIENT_NUM) if masks[c][col])
            assert actual == expected_count, (split_name, col, actual, expected_count)

        tracker = CommTracker(CLIENT_NUM)
        active = list(range(CLIENT_NUM))
        for round_idx in range(ROUNDS):
            tracker.record_round(round_idx, active, masks)
        summary = tracker.summary()
        assert summary['federation_totals']['total_bytes'] == 56_067_340_800, (
            split_name, summary['federation_totals']['total_bytes'])
        assert summary['dtype_bytes'] == DEFAULT_DTYPE_BYTES
        for c in range(CLIENT_NUM):
            assert summary['per_client'][c]['rounds_participated'] == ROUNDS
        print('{}: total_bytes == 56,067,340,800 -- OK'.format(split_name))

    # --- Partial participation: inactive clients must not be touched -------
    tracker = CommTracker(CLIENT_NUM)
    masks_a = _masks_from_pool_sizes(SPLIT_A_POOLS, CLIENT_NUM)
    for round_idx in range(5):
        tracker.record_round(round_idx, list(range(CLIENT_NUM)), masks_a)
    active_this_round = [0, 3, 6]
    tracker.record_round(5, active_this_round, masks_a)
    summary = tracker.summary()
    for c in range(CLIENT_NUM):
        if c in active_this_round:
            assert summary['per_client'][c]['rounds_participated'] == 6
            assert 5 in tracker.uploaded_params[c]
            assert c in summary['per_round'][5]
        else:
            assert summary['per_client'][c]['rounds_participated'] == 5, c
            assert 5 not in tracker.uploaded_params[c], 'inactive client must have no entry for the round'
            assert 5 not in tracker.downloaded_params[c]
            assert c not in summary['per_round'][5]
    print('Partial participation: inactive clients unaffected -- OK')

    # --- CommTracker: single round, no participation at all ----------------
    empty_tracker = CommTracker(4)
    empty_tracker.record_round(0, [], {})
    empty_summary = empty_tracker.summary()
    assert empty_summary['federation_totals']['total_bytes'] == 0
    assert empty_summary['per_round'] == {}
    for c in range(4):
        assert empty_summary['per_client'][c]['rounds_participated'] == 0

    # --- state_dict / load_state_dict round trip ----------------------------
    tracker2 = CommTracker(CLIENT_NUM)
    tracker2.record_round(0, [0, 1], masks_a)
    tracker2.record_round(1, [1, 2], masks_a)
    state = tracker2.state_dict()
    # Simulate a checkpoint round trip through JSON (string round/client keys).
    import json
    reloaded_state = json.loads(json.dumps(state))
    restored = CommTracker(CLIENT_NUM)
    restored.load_state_dict(reloaded_state)
    assert restored.summary() == tracker2.summary()
    assert all(isinstance(k, int) for k in restored.uploaded_params)
    assert all(isinstance(rk, int) for v in restored.uploaded_params.values() for rk in v)

    # --- bytes_to_target_dice ------------------------------------------------
    cumulative_bytes = {0: 1000, 1: 2000, 2: 3000, 3: 4000}
    mean_dice = {0: 40.0, 1: 52.0, 2: 58.0, 3: 61.0}
    hits = bytes_to_target_dice(cumulative_bytes, mean_dice, [50, 55, 60, 65])
    assert hits == {'50': 2000, '55': 3000, '60': 4000, '65': None}, hits
    # Edge case: target already met at round 0.
    hits0 = bytes_to_target_dice(cumulative_bytes, mean_dice, [10])
    assert hits0 == {'10': 1000}
    # Edge case: empty round history.
    assert bytes_to_target_dice({}, {}, [50]) == {'50': None}

    # --- parameter_breakdown --------------------------------------------------
    pool_sizes = {'FLAIR': 7, 'T1ce': 2, 'T1': 5, 'T2': 4}
    breakdown = parameter_breakdown(pool_sizes, CLIENT_NUM, adapter_residual_numel=12345)
    assert breakdown['shared_base'] == SHARED_BASE_PARAMS
    assert breakdown['per_encoder'] == ENCODER_PARAMS
    assert breakdown['private_residual_per_client'] == 12345
    assert breakdown['total'] == SHARED_BASE_PARAMS + 4 * ENCODER_PARAMS + CLIENT_NUM * 12345
    # Edge case: zero residual (e.g. adapter disabled).
    breakdown0 = parameter_breakdown(pool_sizes, CLIENT_NUM, adapter_residual_numel=0)
    assert breakdown0['total'] == SHARED_BASE_PARAMS + 4 * ENCODER_PARAMS
    try:
        parameter_breakdown({'FLAIR': CLIENT_NUM + 1}, CLIENT_NUM, 0)
        raise AssertionError('expected pool-size-too-large to raise')
    except AssertionError as e:
        assert 'exceed' in str(e)

    # --- compression_ratio_vs_full_broadcast ----------------------------------
    one_modality_masks = {c: [True, False, False, False] for c in range(CLIENT_NUM)}
    ratio = compression_ratio_vs_full_broadcast(one_modality_masks)
    expected_ratio = (ENCODER_PARAMS + SHARED_BASE_PARAMS) / (4 * ENCODER_PARAMS + SHARED_BASE_PARAMS)
    assert abs(ratio - expected_ratio) < 1e-12
    assert abs(ratio - 0.48) < 0.01, ratio
    full_masks = {c: [True, True, True, True] for c in range(CLIENT_NUM)}
    assert compression_ratio_vs_full_broadcast(full_masks) == 1.0
    mixed_ratio = compression_ratio_vs_full_broadcast(masks_a)
    assert 0.0 < mixed_ratio < 1.0

    print('ALL TESTS PASSED')

#!/usr/bin/env python3
"""
Offline re-scoring entrypoint for PMCS federated checkpoints.

    python evaluate.py \
      --checkpoint runs/splitA/model_files/last.pth \
      --manifest split/generated/data_seed_20260905/splitA/materialized_config.json \
      --data-root /path/to/BraTS2020_npy \
      --partition test \
      --policy configs/metric_policy.yaml \
      --out runs/splitA/report_v2.json \
      [--save-predictions runs/splitA/preds/]

Re-scores an already-trained, already-saved checkpoint with the unified
scoring pipeline in utils/scoring.py -- no training loop, no GPU required
(pass --device cpu), and this file never imports train_federated.py (the
training entrypoint). Every number in the report comes from exactly one
place: utils.scoring.score() for per-case Dice/HD95/NSD/sensitivity/
specificity, utils.communication for bandwidth accounting, utils.
encoder_coverage for per-modality participation, utils.stats for
distributional/paired-significance/fairness summaries, and utils.
report_schema for the final structural validation. No metric is computed
anywhere in this file itself.

With --save-predictions, every case's per-region binary masks (for every
client and for the residual-zeroed model) are written as compressed .npz,
plus the ground truth once per case. --from-predictions then re-scores
purely from those saved masks -- no checkpoint, no data loader, no forward
pass -- so a policy change (e.g. a different min_component_voxels) can be
re-applied in seconds.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.datasets import Brats_test
from dataset.split_io import load_materialized_split
from models.fusion_net import FusionSegNet
from utils import communication as comm_mod
from utils import encoder_coverage as coverage_mod
from utils import report_schema as schema
from utils import scoring
from utils import stats as stats_mod

MODALITY_NAMES = ['FLAIR', 'T1ce', 'T1', 'T2']
ENCODER_ATTRS = ['flair_encoder', 't1ce_encoder', 't1_encoder', 't2_encoder']
REGIONS = scoring.REGIONS


# ---------------------------------------------------------------------------
# Checkpoint / model reconstruction
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    return torch.load(path, map_location='cpu')


def build_client_model(state_dict, num_class=4):
    model = FusionSegNet(num_cls=num_class)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def zero_residual_with_assertion(model):
    """Deep-copies `model` and zeroes every FusionAdapter residual (R_k) on
    the copy only. Asserts the zeroing actually touched a non-empty set of
    tensors and returns their names -- a silent no-match here (e.g. the
    *_residual naming convention having changed) would make every
    residual-zeroed comparison identically equal to the personalised score,
    which must fail loudly rather than silently reporting a zero effect."""
    model_copy = copy.deepcopy(model)
    touched = []
    with torch.no_grad():
        for name, p in model_copy.named_parameters():
            if name.endswith('_residual'):
                p.zero_()
                touched.append(name)
    if not touched:
        raise AssertionError(
            'zero_residual_with_assertion touched ZERO tensors -- the *_residual '
            'naming convention must have changed. Refusing to silently produce a '
            'residual-zeroed model that is identical to the personalised one.')
    return model_copy, touched


def build_global_model(global_encoders, global_decoder_prior, num_class=4):
    """The single canonical global model: all 4 globally-aggregated encoders
    + the shared decoder prior, with R=0 (a freshly-constructed FusionAdapter
    initialises its residual at exactly zero -- see models/fusion_net.py's
    `dual()` -- and global_decoder_prior never contains residual keys, so
    this is R=0 by construction, not by an extra zeroing step)."""
    model = FusionSegNet(num_cls=num_class)
    for attr, state in zip(ENCODER_ATTRS, global_encoders):
        getattr(model, attr).load_state_dict(state)
    model.fusion_decoder.load_state_dict(global_decoder_prior, strict=False)
    residual_names = [n for n, _ in model.named_parameters() if n.endswith('_residual')]
    for n, p in model.named_parameters():
        if n.endswith('_residual'):
            assert torch.all(p == 0), 'global model residual must be zero at construction: ' + n
    if not residual_names:
        raise AssertionError('build_global_model found ZERO residual-named tensors -- naming convention changed.')
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sliding-window inference (identical patch_size/overlap to the live training
# loop's evaluator, so offline and training-time predictions are directly
# comparable)
# ---------------------------------------------------------------------------

def predict_case(model, x, mask, device, patch_size=80):
    """x: [4,H,W,D] float tensor (unbatched). mask: length-4 bool sequence.
    Returns the predicted integer label map [H,W,D] as a numpy array."""
    model = model.to(device)
    x = x.unsqueeze(0).to(device)
    _, C, H, W, Z = x.size()
    mask_t = torch.from_numpy(np.array(mask)).unsqueeze(0).to(device)

    h_cnt = int(np.ceil((H - patch_size) / (patch_size * 0.5)))
    h_idx_list = [h * int(patch_size * 0.5) for h in range(h_cnt)] + [H - patch_size]
    w_cnt = int(np.ceil((W - patch_size) / (patch_size * 0.5)))
    w_idx_list = [w * int(patch_size * 0.5) for w in range(w_cnt)] + [W - patch_size]
    z_cnt = int(np.ceil((Z - 80) / (80 * 0.5)))
    z_idx_list = [z * int(80 * 0.5) for z in range(z_cnt)] + [Z - 80]

    one_tensor = torch.ones(1, patch_size, patch_size, 80).float().to(device)
    weight = torch.zeros(1, 1, H, W, Z).float().to(device)
    for h in h_idx_list:
        for w in w_idx_list:
            for z in z_idx_list:
                weight[:, :, h:h + patch_size, w:w + patch_size, z:z + 80] += one_tensor
    weight = weight.repeat(1, 4, 1, 1, 1)

    pred = torch.zeros(1, 4, H, W, Z).float().to(device)
    model.is_training = False
    with torch.no_grad():
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size, w:w + patch_size, z:z + 80]
                    pred_part, _, _ = model(x_input, mask_t, None, None, None, None)
                    pred[:, :, h:h + patch_size, w:w + patch_size, z:z + 80] += pred_part
    pred = pred / weight
    pred = torch.argmax(pred, dim=1)[0]
    return pred.cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Spacing (Task 4): read from a NIfTI affine when available, else fall back
# to the configured/reported expected_spacing_mm -- never a silent [1,1,1].
# ---------------------------------------------------------------------------

def resolve_spacing_mm(case_id, nifti_root, expected_spacing_mm, tolerance=1e-3):
    """Returns (spacing_mm, source).

    If nifti_root is NOT given at all, uses expected_spacing_mm directly and
    says so -- this is a configured value, not a code-level hardcoded
    constant, and the report always records that no real per-case check was
    performed.

    If nifti_root IS given, a real per-case check was explicitly requested:
    a matching file must be found and its affine must match
    expected_spacing_mm within `tolerance`. Both "no matching file" and "the
    file's spacing doesn't match" raise -- silently falling back to the
    configured default here would defeat the entire point of passing
    --nifti-root, indistinguishable from never having passed it.
    """
    if not nifti_root:
        return tuple(float(v) for v in expected_spacing_mm), 'policy_default (--nifti-root not supplied)'

    import glob
    import nibabel as nib
    candidates = glob.glob(os.path.join(nifti_root, '**', '{}*.nii*'.format(case_id)), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            'case {}: --nifti-root {!r} was supplied but no matching NIfTI file was found under it -- '
            'refusing to silently fall back to the configured expected_spacing_mm for a case that was '
            'supposed to get a real per-case spacing check.'.format(case_id, nifti_root))

    img = nib.load(candidates[0])
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    expected = tuple(float(v) for v in expected_spacing_mm)
    if any(abs(a - b) > tolerance for a, b in zip(spacing, expected)):
        raise ValueError(
            'case {}: observed NIfTI spacing {} does not match configured '
            'expected_spacing_mm {} (tolerance {})'.format(case_id, spacing, expected, tolerance))
    return spacing, 'nifti_affine:{}'.format(candidates[0])


# ---------------------------------------------------------------------------
# Prediction save / load (--save-predictions / --from-predictions)
# ---------------------------------------------------------------------------

def save_entity_masks(save_dir, entity_id, case_id, masks):
    entity_dir = os.path.join(save_dir, entity_id)
    os.makedirs(entity_dir, exist_ok=True)
    path = os.path.join(entity_dir, case_id + '.npz')
    stacked = np.stack([masks[r].astype(np.uint8) for r in REGIONS], axis=0)
    np.savez_compressed(path, masks=stacked, regions=np.array(REGIONS))


def load_entity_masks(save_dir, entity_id, case_id):
    path = os.path.join(save_dir, entity_id, case_id + '.npz')
    data = np.load(path)
    stacked = data['masks'].astype(bool)
    regions = list(data['regions'])
    return {r: stacked[regions.index(r)] for r in REGIONS}


def save_residual_zeroing_log(save_dir, residual_touch_log):
    """Persists the evidence that residual zeroing actually touched a
    non-empty set of tensors for every client, alongside the saved
    prediction masks, so a later --from-predictions re-score can carry that
    evidence forward instead of reporting an empty (and therefore
    schema-invalid) residual_zeroing_log."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'residual_zeroing_log.json'), 'w') as f:
        json.dump(residual_touch_log, f, indent=2)


def load_residual_zeroing_log(save_dir):
    path = os.path.join(save_dir, 'residual_zeroing_log.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            '--from-predictions {} has no residual_zeroing_log.json -- it was not created by a prior '
            '--save-predictions run of this evaluate.py, so there is no evidence that residual zeroing '
            'ever touched a non-empty set of tensors for these saved masks. Refusing to assemble a report '
            'with a fabricated or empty residual_zeroing_log.'.format(save_dir))
    with open(path, 'r') as f:
        log = json.load(f)
    if not log or not all(v for v in log.values()):
        raise ValueError(
            '{} exists but is empty or has an entity with zero zeroed tensors -- refusing to '
            'assemble a report from it.'.format(path))
    return log


# ---------------------------------------------------------------------------
# Provenance (Task 8)
# ---------------------------------------------------------------------------

def _git_info():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        commit, dirty = None, None
    return commit, dirty


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_provenance(args, resolved_config, wall_clock_seconds=None):
    commit, dirty = _git_info()
    versions = {}
    for mod_name in ('torch', 'numpy', 'scipy', 'medpy', 'nibabel'):
        try:
            versions[mod_name] = __import__(mod_name).__version__
        except Exception:
            versions[mod_name] = None

    is_cuda = str(args.device).startswith('cuda')
    gpu_hours = (wall_clock_seconds / 3600.0) if (is_cuda and wall_clock_seconds is not None) else 0.0

    return {
        'git_commit': commit,
        'git_dirty': dirty,
        'config': resolved_config,
        'manifest_path': os.path.abspath(args.manifest) if args.manifest else None,
        'manifest_sha256': _file_sha256(args.manifest) if args.manifest and os.path.exists(args.manifest) else None,
        'library_versions': versions,
        'python_version': sys.version,
        'platform': platform.platform(),
        'device': args.device,
        'seed': args.seed,
        'wall_clock_seconds': wall_clock_seconds,
        # GPU-hours is wall-clock time attributed to the run only when it
        # actually ran on a GPU (--device cuda*); a CPU run reports 0.0, not
        # None, since "how much GPU time did this cost" has a real, exact
        # answer (zero) rather than being undefined.
        'gpu_hours': gpu_hours,
    }


def set_determinism(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        print('WARNING: torch.use_deterministic_algorithms(True) could not be fully enabled: {}'.format(exc),
              file=sys.stderr)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Communication / encoder-coverage reconstruction from static split facts
# ---------------------------------------------------------------------------

def reconstruct_federation_history(masks, rounds, active_clients_by_round=None):
    """masks: list of length-4 bool masks, one per client (index == client_idx).
    rounds: total round count. active_clients_by_round: optional
    {round_idx: [client_idx,...]}; when omitted, EVERY client is assumed
    active EVERY round (full participation) -- true for both completed
    Split A/B runs -- and the report records that this was an assumption,
    not a measurement, so it is never mistaken for a per-round log.
    """
    client_num = len(masks)
    masks_dict = {c: masks[c] for c in range(client_num)}
    assumed_full = active_clients_by_round is None

    comm_tracker = comm_mod.CommTracker(client_num)
    coverage_tracker = coverage_mod.EncoderCoverageTracker(MODALITY_NAMES)
    pool_sizes = {name: sum(1 for m in masks if m[i]) for i, name in enumerate(MODALITY_NAMES)}

    for r in range(rounds):
        active = list(range(client_num)) if assumed_full else active_clients_by_round.get(r, [])
        comm_tracker.record_round(r, active, masks_dict)
        contributor_counts = [sum(1 for c in active if masks[c][m]) for m in range(4)]
        coverage_tracker.record_round(r, contributor_counts)

    comm_summary = comm_tracker.summary()
    comm_summary['participation_assumed_full'] = assumed_full
    comm_summary['parameter_breakdown'] = comm_mod.parameter_breakdown(
        pool_sizes, client_num, adapter_residual_numel=_ADAPTER_RESIDUAL_NUMEL)
    comm_summary['compression_ratio_vs_full_broadcast'] = comm_mod.compression_ratio_vs_full_broadcast(masks_dict)

    coverage_summary = coverage_mod.summarize(
        coverage_tracker, pool_sizes, total_clients=client_num,
        participation_k=(client_num if assumed_full else None))
    coverage_summary['participation_assumed_full'] = assumed_full
    return comm_summary, coverage_summary


_ADAPTER_RESIDUAL_NUMEL = None  # filled in once a real model is instantiated


# ---------------------------------------------------------------------------
# Per-partition scoring
# ---------------------------------------------------------------------------

def score_case_list(entity_id, model, mask, loader, device, policy, nifti_root, save_dir, from_predictions):
    """Runs (or loads) predictions for every case in `loader` and scores each
    with utils.scoring.score(). Returns a list of per-case result dicts, each
    {'case_id':..., 'scores': score()-shaped dict, 'spacing_mm':..., 'spacing_source':...}."""
    results = []
    for x, y, name in loader:
        case_id = name[0] if isinstance(name, (list, tuple)) else name

        spacing_mm, spacing_source = resolve_spacing_mm(case_id, nifti_root, policy['expected_spacing_mm'])

        if from_predictions:
            # y is None here (see _FakeLoaderFromCaseIds) -- ground truth
            # comes from the saved gt masks instead, never from a loader.
            pred_masks = load_entity_masks(from_predictions, entity_id, case_id)
            gt_masks = load_entity_masks(from_predictions, 'gt', case_id)
            case_scores = scoring.score(pred_masks, gt_masks, spacing_mm, policy)
        else:
            gt_label = y[0].numpy() if y.dim() == 4 else y.numpy()
            pred_label = predict_case(model, x[0], mask, device)
            case_scores = scoring.score(pred_label, gt_label, spacing_mm, policy)
            if save_dir:
                save_entity_masks(save_dir, entity_id, case_id, scoring.region_masks(pred_label))
                gt_npz_path = os.path.join(save_dir, 'gt', case_id + '.npz')
                if not os.path.exists(gt_npz_path):
                    save_entity_masks(save_dir, 'gt', case_id, scoring.region_masks(gt_label))

        results.append({'case_id': case_id, 'scores': case_scores,
                         'spacing_mm': list(spacing_mm), 'spacing_source': spacing_source})
    return results


def aggregate_region_field(per_case_results, region, field):
    """Pulls one scalar field (e.g. 'dice_raw') for one region out of a list
    of per-case score() outputs, in case order."""
    return [c['scores'][region][field] for c in per_case_results]


def build_region_matrix(per_client_case_results, field):
    """{client_idx: [mean_WT, mean_TC, mean_ET]} using distribution_stats' mean
    over that client's cases for `field` (e.g. 'dice_raw')."""
    matrix = {}
    for c, case_results in per_client_case_results.items():
        row = []
        for region in REGIONS:
            values = aggregate_region_field(case_results, region, field)
            row.append(stats_mod.distribution_stats(values)['mean'])
        matrix[c] = row
    return matrix


# ---------------------------------------------------------------------------
# Report assembly (Task 1: "no metric may be computed anywhere else" --
# everything below only *reads* results already produced by score()/the
# communication/coverage/stats modules and arranges them into the schema)
# ---------------------------------------------------------------------------

def assemble_report(args, policy, masks, client_modalities, rounds,
                     per_client_case_results, per_client_residual_zeroed_results,
                     global_model_case_results, round_selection, wall_clock_seconds=None,
                     per_client_on_global_test_results=None):
    partition = args.partition
    client_num = len(masks)

    dice_matrix = build_region_matrix(per_client_case_results, 'dice_raw')
    dice_postproc_matrix = build_region_matrix(per_client_case_results, 'dice_postproc')
    hd95_matrix = build_region_matrix(per_client_case_results, 'hd95_raw')
    hd95_postproc_matrix = build_region_matrix(per_client_case_results, 'hd95_postproc')
    hd95_valid_pairs_only_matrix = {
        c: [stats_mod.distribution_stats(
            aggregate_region_field(rows, region, 'hd95_raw_valid_pairs_only'))['mean'] for region in REGIONS]
        for c, rows in per_client_case_results.items()
    }

    residual_zeroed_matrix = build_region_matrix(per_client_residual_zeroed_results, 'dice_raw')
    residual_zeroed_minus_personalised = {
        c: [residual_zeroed_matrix[c][i] - dice_matrix[c][i] if residual_zeroed_matrix[c][i] is not None
            and dice_matrix[c][i] is not None else None
            for i in range(len(REGIONS))]
        for c in range(client_num)
    }
    personalisation_gain = {
        c: [-v if v is not None else None for v in residual_zeroed_minus_personalised[c]]
        for c in range(client_num)
    }

    per_case_records = []
    for c, rows in per_client_case_results.items():
        for row in rows:
            for region in REGIONS:
                s = row['scores'][region]
                per_case_records.append({
                    'entity': 'client_{}'.format(c), 'partition': partition, 'case_id': row['case_id'],
                    'region': region, 'spacing_mm': row['spacing_mm'], 'spacing_source': row['spacing_source'],
                    **s,
                })
    # Residual-zeroed per-case records: without these, per-case
    # personalisation gain (residual-zeroed vs personalised, matched by
    # case_id) cannot be computed from the final per-case table at all --
    # only the pre-aggregated mean-based residual_zeroed_minus_personalised
    # matrix would be available.
    for c, rows in per_client_residual_zeroed_results.items():
        for row in rows:
            for region in REGIONS:
                s = row['scores'][region]
                per_case_records.append({
                    'entity': 'client_{}_residual_zeroed'.format(c), 'partition': partition,
                    'case_id': row['case_id'], 'region': region, 'spacing_mm': row['spacing_mm'],
                    'spacing_source': row['spacing_source'],
                    **s,
                })
    for row in global_model_case_results:
        for region in REGIONS:
            s = row['scores'][region]
            per_case_records.append({
                'entity': 'global_model', 'partition': 'global_held_out_test', 'case_id': row['case_id'],
                'region': region, 'spacing_mm': row['spacing_mm'], 'spacing_source': row['spacing_source'],
                **s,
            })

    # Each client's own personalised model scored on the SAME shared
    # 50-case held-out set as every other client and the global model --
    # unlike per_client_case_results (each client's own, disjoint, ~6-case
    # partition), these rows share case_id across clients, which is what
    # makes a paired per-case comparison (e.g. Wilcoxon between two
    # clients' Dice on the identical 50 cases) possible at all.
    client_on_global_test_matrix = {}
    per_client_on_global_test_results = per_client_on_global_test_results or {}
    for c, rows in per_client_on_global_test_results.items():
        for row in rows:
            for region in REGIONS:
                s = row['scores'][region]
                per_case_records.append({
                    'entity': 'client_{}_on_global_test'.format(c), 'partition': 'global_held_out_test',
                    'case_id': row['case_id'], 'region': region, 'spacing_mm': row['spacing_mm'],
                    'spacing_source': row['spacing_source'],
                    **s,
                })
        client_on_global_test_matrix[c] = [
            stats_mod.distribution_stats(aggregate_region_field(rows, region, 'dice_raw'))['mean']
            for region in REGIONS
        ]

    per_case_stats = {}
    for entity_key, rows in list(per_client_case_results.items()) + [('global', global_model_case_results)]:
        entity_name = 'client_{}'.format(entity_key) if isinstance(entity_key, int) else entity_key
        per_case_stats[entity_name] = {
            region: {
                metric: stats_mod.distribution_stats(aggregate_region_field(rows, region, metric))
                for metric in ('dice_raw', 'dice_postproc', 'hd95_raw', 'hd95_postproc')
            } for region in REGIONS
        }

    per_client_mean_dice_pct = {
        c: float(np.mean([v for v in dice_matrix[c] if v is not None])) * 100.0 for c in range(client_num)
    }
    fairness = stats_mod.fairness_block(per_client_mean_dice_pct, client_modalities, MODALITY_NAMES)

    comm_summary, coverage_summary = reconstruct_federation_history(masks, rounds)

    evaluated_rounds = [rounds - 1]
    rounds_to_target = None
    if schema.should_emit_rounds_to_target(evaluated_rounds):
        rounds_to_target = {}  # only ever populated when >1 round was actually evaluated here

    postproc_policy = policy['postproc']
    hd95_policy = {
        'implementation': 'medpy.metric.binary.hd95',
        'both_empty': 0.0,
        'mismatch_empty': 'physical_image_diagonal',
        'spacing_mm_by_case': 'see per_case_metrics; not a single global constant',
    }

    report = {
        'schema_version': schema.SCHEMA_VERSION,
        'accounting_version': schema.ACCOUNTING_VERSION,
        'evaluation_partition': partition,
        'round_selection': round_selection,
        'communication': comm_summary,
        'encoder_coverage': coverage_summary,
        'postproc_policy': postproc_policy,
        'hd95_policy': hd95_policy,
        'client_modalities': {c: masks[c] for c in range(client_num)},
        'provenance': build_provenance(
            args, {'policy_path': os.path.abspath(args.policy) if args.policy else None},
            wall_clock_seconds=wall_clock_seconds),
        schema.partitioned_key('dice_matrix', partition): dice_matrix,
        schema.partitioned_key('dice_postproc_matrix', partition): dice_postproc_matrix,
        schema.partitioned_key('hd95_matrix', partition): hd95_matrix,
        schema.partitioned_key('hd95_postproc_matrix', partition): hd95_postproc_matrix,
        schema.partitioned_key('hd95_valid_pairs_only_matrix', partition): hd95_valid_pairs_only_matrix,
        schema.partitioned_key('personalisation_gain_matrix', partition): personalisation_gain,
        'residual_zeroed_minus_personalised': residual_zeroed_minus_personalised,
        'per_case_stats': per_case_stats,
        'fairness': fairness,
        # Each client's own personalised model, scored on the same shared
        # 50-case global_held_out_test set the canonical global model uses.
        # Because every client shares the identical 50 case_ids here (unlike
        # dice_matrix, where each client has its own disjoint partition),
        # per_case_records for entity "client_{k}_on_global_test" can be
        # paired by case_id across any two clients and fed directly into
        # utils.stats.wilcoxon_paired for a same-sample significance test --
        # that pairing is impossible from dice_matrix/per_client_case_results
        # alone, since no two clients evaluate the same cases there.
        'client_on_global_test_dice_matrix': client_on_global_test_matrix,
        'global_model': {
            'dice': stats_mod.distribution_stats(
                [np.mean([row['scores'][r]['dice_raw'] for r in REGIONS]) for row in global_model_case_results]),
            'per_region': {
                region: {
                    'dice_raw': stats_mod.distribution_stats(aggregate_region_field(global_model_case_results, region, 'dice_raw')),
                    'dice_postproc': stats_mod.distribution_stats(aggregate_region_field(global_model_case_results, region, 'dice_postproc')),
                    'hd95_raw': stats_mod.distribution_stats(aggregate_region_field(global_model_case_results, region, 'hd95_raw')),
                    'hd95_postproc': stats_mod.distribution_stats(aggregate_region_field(global_model_case_results, region, 'hd95_postproc')),
                } for region in REGIONS
            },
            'n_cases': len(global_model_case_results),
            'partition': 'global_held_out_test',
            'private_residual_zeroed': True,
        },
    }
    if rounds_to_target is not None:
        report[schema.partitioned_key('rounds_to_target_dice', partition)] = rounds_to_target

    schema.add_deprecated_alias(report, 'residual_zeroed_minus_personalised', 'test_global_minus_client')
    schema.add_deprecated_alias(
        report, schema.partitioned_key('hd95_valid_pairs_only_matrix', partition), 'test_hd95_valid_pairs_only_matrix')

    return report, per_case_records


def to_jsonable(obj):
    """Recursively converts numpy/torch scalars to plain JSON-safe Python,
    and normalises any non-finite float to None as a general safety net
    (nothing currently produced by this pipeline is expected to be
    non-finite -- encoder_coverage's effective_horizon = T * (1 -
    idle_fraction) is always a finite number -- but a NaN/Inf slipping in
    anywhere must never silently reach the JSON output). report_schema.
    validate_report() treats every field as either a finite number or an
    explicit "not applicable" None -- there is no JSON representation of
    infinity, so any non-finite value becomes None at this boundary; this
    only affects what gets written to
    disk."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_per_case_csv(path, records):
    import csv
    if not records:
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(to_jsonable(v)) if isinstance(v, (dict, list)) else v
                              for k, v in r.items()})


def write_per_case_parquet(path, records):
    """The primary per-case artifact (Task 7): every raw and post-processed
    metric, per case, per region, per entity -- never collapsed to a mean
    before it reaches disk. Nested fields (edge_counts, nsd_raw/postproc,
    spacing_mm) are stored as native parquet struct/list columns rather than
    JSON-stringified, unlike the CSV companion, since parquet's columnar
    format handles them natively and this keeps them queryable without a
    JSON-parsing step."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        # Still write an (empty) file rather than silently producing
        # nothing -- a downstream reader expecting this path to exist
        # should see zero rows, not a missing file.
        pq.write_table(pa.table({}), path)
        return
    table = pa.Table.from_pylist(to_jsonable(records))
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Acceptance-test helper: diff against a previously published report
# ---------------------------------------------------------------------------

def find_old_matrix(old_report, old_key):
    """Finds `old_key` (e.g. 'test_dice_matrix') in a previously published
    report, trying two shapes: a flat top-level key (this module's own
    report_v2.json schema), or nested inside rounds[<round_idx>] (the live
    training-loop's metrics.json schema, which never has it at the top
    level). Returns the raw matrix (list-of-rows or dict), or None if
    neither shape has it anywhere -- the caller (--compare-against) treats
    None as a hard failure of the mandatory reproduction gate, never as
    "nothing to compare, carry on"."""
    old_matrix_raw = old_report.get(old_key)
    if old_matrix_raw is not None or 'rounds' not in old_report:
        return old_matrix_raw

    for round_idx in sorted((int(k) for k in old_report['rounds'].keys()), reverse=True):
        payload = old_report['rounds'][str(round_idx)]
        if old_key in payload:
            return payload[old_key]
    return None


def print_diff_and_check(new_matrix, old_matrix, tol=1e-4, label='dice_matrix'):
    print('\n{} diff (new vs old), tolerance {}:'.format(label, tol))
    print('{:<8}{:>10}{:>10}{:>10}   {:>10}{:>10}{:>10}   {:>8}'.format(
        'client', 'WT_new', 'TC_new', 'ET_new', 'WT_old', 'TC_old', 'ET_old', 'max_diff'))
    ok = True
    for c in sorted(int(k) for k in old_matrix.keys()):
        old_row = old_matrix[str(c)] if str(c) in old_matrix else old_matrix[c]
        new_row = new_matrix.get(c, new_matrix.get(str(c)))
        diffs = [abs(a - b) for a, b in zip(new_row, old_row)]
        max_diff = max(diffs)
        ok = ok and max_diff <= tol
        print('{:<8}{:>10.4f}{:>10.4f}{:>10.4f}   {:>10.4f}{:>10.4f}{:>10.4f}   {:>8.5f}'.format(
            c, *new_row, *old_row, max_diff))
    print('RESULT: {}'.format('PASS' if ok else 'FAIL'))
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', help='path to the consolidated last.pth (required unless --from-predictions)')
    p.add_argument('--manifest', required=True, help='materialized split config JSON (masks + per-client file lists)')
    p.add_argument('--data-root', help='BraTS npy root (required unless --from-predictions)')
    p.add_argument('--partition', required=True, choices=('validation', 'test'))
    p.add_argument('--policy', required=True, help='configs/metric_policy.yaml')
    p.add_argument('--out', required=True, help='output report_v2.json path')
    p.add_argument('--save-predictions', default=None)
    p.add_argument('--from-predictions', default=None,
                    help='re-score from a directory previously written by --save-predictions; no forward pass')
    p.add_argument('--nifti-root', default=None, help='optional: source NIfTI directory for a real per-case spacing check')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-seed', type=int, required=True)
    p.add_argument('--rounds', type=int, default=None,
                    help='total federated rounds (default: read from checkpoint[\'round\'])')
    p.add_argument('--round-selection', default='last_round', choices=('last_round', 'best_validation'))
    p.add_argument('--best-checkpoint-dir', default=None,
                    help='directory containing client-*_model_best.pth (default: --checkpoint\'s directory)')
    p.add_argument('--compare-against', default=None,
                    help='an existing report/metrics.json to diff this run\'s dice matrix against (acceptance test)')
    p.add_argument('--compare-tolerance', type=float, default=1e-4)
    return p.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    set_determinism(args.seed)

    policy = scoring.load_policy(args.policy)
    # Print the RESOLVED policy -- the actual dict that will be passed into
    # every score() call -- not just the path to the file on disk. This is
    # the single fastest way to tell "the legacy/none policy was used
    # (correct, no bug)" apart from "the postproc branch never fired (bug)":
    # if this print ever shows method: none for every region, dice_postproc
    # == dice_raw is the CORRECT, expected result, not a defect.
    print('Resolved postproc policy (--policy {}):'.format(args.policy))
    print(json.dumps(policy['postproc'], indent=2))
    active_methods = {region: policy['postproc'][region]['method'] for region in scoring.REGIONS}
    if all(m == 'none' for m in active_methods.values()):
        print('All three regions resolve to method=none: dice_postproc == dice_raw is EXPECTED here, not a bug.')
    else:
        print('At least one region has an active method {} -- dice_postproc should differ from '
              'dice_raw whenever that region\'s post-processing has anything to do.'.format(active_methods))

    masks, train_files, validation_files, test_files, global_test_file, split_metadata = load_materialized_split(
        args.manifest, client_num=8, data_seed=args.data_seed)
    client_num = len(masks)
    client_modalities = {c: masks[c] for c in range(client_num)}
    partition_files = validation_files if args.partition == 'validation' else test_files

    device = torch.device(args.device)

    global _ADAPTER_RESIDUAL_NUMEL
    template = FusionSegNet(num_cls=4)
    _ADAPTER_RESIDUAL_NUMEL = sum(
        p.numel() for n, p in template.fusion_decoder.adapter.named_parameters() if n.endswith('_residual'))
    del template

    ckpt = None
    rounds = args.rounds
    if not args.from_predictions:
        if not args.checkpoint or not args.data_root:
            raise ValueError('--checkpoint and --data-root are required unless --from-predictions is used')
        ckpt = load_checkpoint(args.checkpoint)
        if rounds is None:
            rounds = int(ckpt['round'])
    if rounds is None:
        raise ValueError('--rounds must be given when using --from-predictions without a checkpoint')

    round_selection = {'mode': args.round_selection, 'round_index': rounds - 1}

    per_client_case_results = {}
    per_client_residual_zeroed_results = {}
    per_client_on_global_test_results = {}
    residual_touch_log = {}

    if args.from_predictions:
        # No model is loaded in this mode, so no fresh residual-zeroing
        # assertion runs here -- the evidence must be the one recorded when
        # --save-predictions originally created these masks (with a real
        # model, going through zero_residual_with_assertion). Load it rather
        # than leaving residual_zeroing_log empty, which would make a
        # from-predictions report indistinguishable from one where zeroing
        # silently matched nothing.
        residual_touch_log = load_residual_zeroing_log(args.from_predictions)

    # Built once, reused by every client's own-mask evaluation below AND by
    # the canonical global model further down -- a DataLoader/fake loader is
    # freely re-iterable, and building the 50-case loader once instead of 9
    # times avoids repeatedly re-globbing --data-root for the same files.
    global_loader = None
    global_model = None
    if global_test_file:
        if args.from_predictions:
            global_loader = _FakeLoaderFromCaseIds(_read_case_ids_only(global_test_file))
        else:
            global_loader = DataLoader(
                Brats_test(transforms='Compose([NumpyType((np.float32, np.int64)),])', root=args.data_root,
                           modal='all', test_file=global_test_file, all_=True),
                batch_size=1, shuffle=False, num_workers=0)
            global_model = build_global_model(ckpt['global_encoders'], ckpt['global_decoder_prior'])

    for c in range(client_num):
        model = None
        residual_zeroed_model = None
        if args.from_predictions:
            # Case list still comes from the manifest's file, without ever
            # touching --data-root or running a forward pass.
            loader = _FakeLoaderFromCaseIds(_read_case_ids_only(partition_files[c + 1]))
        else:
            loader = DataLoader(
                Brats_test(transforms='Compose([NumpyType((np.float32, np.int64)),])', root=args.data_root,
                           modal='all', test_file=partition_files[c + 1], all_=True),
                batch_size=1, shuffle=False, num_workers=0)
            client_state = ckpt['clients_dict'][c]
            model = build_client_model(client_state)
            residual_zeroed_model, touched = zero_residual_with_assertion(model)
            residual_touch_log['client_{}'.format(c)] = touched

        per_client_case_results[c] = score_case_list(
            'client_{}'.format(c), model, masks[c], loader, device, policy, args.nifti_root,
            args.save_predictions, args.from_predictions)
        per_client_residual_zeroed_results[c] = score_case_list(
            'client_{}_residual_zeroed'.format(c), residual_zeroed_model, masks[c], loader, device, policy,
            args.nifti_root, args.save_predictions, args.from_predictions)

        # Each client's own personalised model, additionally scored on the
        # SAME shared 50-case held-out set every other client and the global
        # model are scored on (using this client's own mask, not full
        # modalities -- this is "how does this client's actual deployed
        # model perform on unseen shared patients", not a hypothetical
        # full-modality variant of it). Without this, every client's Dice is
        # only ever measured on its own ~6-case partition, and per-case
        # paired comparisons (e.g. Wilcoxon between two clients) are
        # impossible since no two clients share any evaluated cases at all.
        if global_loader is not None:
            per_client_on_global_test_results[c] = score_case_list(
                'client_{}_on_global_test'.format(c), model, masks[c], global_loader, device, policy,
                args.nifti_root, args.save_predictions, args.from_predictions)

    # Canonical global model, scored on the same 50-case held-out set, with
    # all four modalities (every case has all four in the data).
    global_model_case_results = []
    if global_loader is not None:
        global_model_case_results = score_case_list(
            'global_model', global_model, [True, True, True, True], global_loader, device, policy,
            args.nifti_root, args.save_predictions, args.from_predictions)

    if not args.from_predictions and args.save_predictions:
        save_residual_zeroing_log(args.save_predictions, residual_touch_log)

    report, per_case_records = assemble_report(
        args, policy, masks, client_modalities, rounds,
        per_client_case_results, per_client_residual_zeroed_results,
        global_model_case_results, round_selection, wall_clock_seconds=time.time() - start_time,
        per_client_on_global_test_results=per_client_on_global_test_results)
    report['split_metadata'] = split_metadata
    report['residual_zeroing_log'] = residual_touch_log

    report = to_jsonable(report)
    schema.validate_report(report)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    per_case_parquet_path = os.path.join(out_dir, 'per_case_metrics.parquet')
    write_per_case_parquet(per_case_parquet_path, per_case_records)
    per_case_csv_path = os.path.join(out_dir, 'per_case_metrics.csv')
    write_per_case_csv(per_case_csv_path, per_case_records)

    print('Wrote {}'.format(args.out))
    print('Wrote {}'.format(per_case_parquet_path))
    print('Wrote {}'.format(per_case_csv_path))

    if args.compare_against:
        with open(args.compare_against) as f:
            old_report = json.load(f)
        old_key = 'test_dice_matrix' if args.partition == 'test' else 'validation_dice_matrix'
        new_key = schema.partitioned_key('dice_matrix', args.partition)

        old_matrix_raw = find_old_matrix(old_report, old_key)
        if old_matrix_raw is None:
            # --compare-against is a mandatory reproduction gate, not an
            # advisory check: if the caller asked for a comparison and no
            # baseline can be found to compare against, that is a failure of
            # the gate itself, not something to warn about and continue past.
            print('ERROR: {} has no {} at its top level or inside any rounds[*] entry -- '
                  'cannot perform the mandatory reproduction check.'.format(args.compare_against, old_key),
                  file=sys.stderr)
            sys.exit(1)

        old_matrix = {i: row for i, row in enumerate(old_matrix_raw)} \
            if isinstance(old_matrix_raw, list) else old_matrix_raw
        ok = print_diff_and_check(report[new_key], old_matrix, tol=args.compare_tolerance, label=new_key)
        if not ok:
            sys.exit(1)


def _read_case_ids_only(path):
    with open(path, 'r') as handle:
        return [line.strip() for line in handle if line.strip()]


class _FakeLoaderFromCaseIds:
    """Used only in --from-predictions mode: yields (None, None, [case_id])
    tuples so score_case_list's loop shape matches the real DataLoader's,
    without ever touching --data-root or running a forward pass."""
    def __init__(self, case_ids):
        self.case_ids = case_ids

    def __iter__(self):
        for cid in self.case_ids:
            yield None, None, [cid]


if __name__ == '__main__':
    main()

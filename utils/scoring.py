"""
Unified scoring module for PMCS BraTS evaluation.

The whole point of this file is that there is exactly ONE code path from
(prediction, ground truth) -> metrics, and every entity -- each of the 8
personalised clients as well as the residual-zeroed/global model -- goes
through it identically. `score()` never knows or cares "who" it is scoring;
it only ever sees arrays. This fixes the previous defect where the global
model silently got extra post-processed Dice/HD95 numbers that the clients
never received, making the two incomparable.

Label convention (matches utils/metrics.py / utils/predict.py):
    0 = background, 1 = NET (necrotic core), 2 = ED (edema), 3 = ET
Regions derived from those labels:
    WT (whole tumour) = {1,2,3}
    TC (tumour core)  = {1,3}
    ET (enhancing)    = {3}
"""
import numpy as np
from scipy import ndimage

REGIONS = ('WT', 'TC', 'ET')

POSTPROC_METHODS = ('none', 'largest_cc', 'size_filter')

DEFAULT_POLICY = {
    'postproc': {
        'WT': {'method': 'largest_cc', 'min_component_voxels': 50},
        'TC': {'method': 'size_filter', 'min_component_voxels': 50},
        'ET': {'method': 'size_filter', 'min_component_voxels': 50},
        'et_volume_threshold': 500,
        'enforce_nesting': True,
    },
    'expected_spacing_mm': [1.0, 1.0, 1.0],
    'nsd_tolerances_mm': [1.0, 3.0],
}


# ---------------------------------------------------------------------------
# Region derivation
# ---------------------------------------------------------------------------

def region_masks(label_map):
    """label_map: integer ndarray with values in {0,1,2,3} (NET=1, ED=2, ET=3,
    BG=0). Returns {'WT','TC','ET'} bool ndarrays derived per the BraTS region
    convention. WT/TC/ET are automatically nested (ET subset TC subset WT) in
    the raw label map by construction -- nesting can only be broken later, by
    independent per-region post-processing (see enforce_nesting)."""
    label_map = np.asarray(label_map)
    net = label_map == 1
    ed = label_map == 2
    et = label_map == 3
    return {
        'WT': net | ed | et,
        'TC': net | et,
        'ET': et,
    }


# ---------------------------------------------------------------------------
# Policy loading / validation
# ---------------------------------------------------------------------------

def load_policy(path):
    """Loads and validates the YAML metric policy file at `path`, returning a
    plain dict. Raises ValueError if any postproc method for WT/TC/ET is not
    one of POSTPROC_METHODS."""
    import yaml

    with open(path, 'r') as f:
        policy = yaml.safe_load(f)
    _validate_policy(policy)
    return policy


def _validate_policy(policy):
    postproc = policy.get('postproc', {})
    for region in REGIONS:
        region_cfg = postproc.get(region, {})
        method = region_cfg.get('method')
        if method not in POSTPROC_METHODS:
            raise ValueError(
                "policy postproc[{!r}].method must be one of {!r}, got {!r}".format(
                    region, POSTPROC_METHODS, method))
    return policy


# ---------------------------------------------------------------------------
# Post-processing (predictions only -- never applied to ground truth)
# ---------------------------------------------------------------------------

def apply_region_postproc(mask, method, min_component_voxels=50):
    """mask: bool ndarray (any shape, assumed 3D). `method`:
      'none'       -> unchanged.
      'largest_cc' -> keep only the single largest connected component
                      (6-connectivity, scipy.ndimage.label default structure).
      'size_filter'-> keep every connected component with >= min_component_voxels
                      voxels, drop the rest. Deliberately NOT "keep only the
                      largest": a multifocal or ring-shaped enhancing tumour
                      can have multiple genuinely-separate true-positive
                      components, so TC/ET use size_filter, never largest_cc.
    Ground truth must never be passed through this function."""
    if method not in POSTPROC_METHODS:
        raise ValueError('unknown postproc method: {!r}'.format(method))
    if method == 'none' or not mask.any():
        return mask

    labeled, num_components = ndimage.label(mask)
    if num_components <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num_components + 1))

    if method == 'largest_cc':
        keep_labels = [int(np.argmax(sizes)) + 1]
    else:  # size_filter
        keep_labels = [i + 1 for i, size in enumerate(sizes) if size >= min_component_voxels]

    if not keep_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_labels)


def apply_et_volume_threshold(et_mask, et_volume_threshold):
    """If et_volume_threshold > 0 and et_mask has fewer true voxels than that,
    return an all-False mask of the same shape (the classic BraTS
    "suppress tiny ET" heuristic). et_volume_threshold <= 0 disables the
    feature (no-op, mask returned unchanged)."""
    if et_volume_threshold > 0 and int(et_mask.sum()) < et_volume_threshold:
        return np.zeros_like(et_mask, dtype=bool)
    return et_mask


def enforce_nesting(masks):
    """masks: {'WT','TC','ET'} bool ndarrays that may have been independently
    post-processed and may therefore violate ET subset TC subset WT. Re-derives
    a nested-consistent version by cascading AND from the top down:
        WT_final = WT (unchanged)
        TC_final = TC AND WT_final
        ET_final = ET AND TC_final
    """
    wt = masks['WT']
    tc_final = masks['TC'] & wt
    et_final = masks['ET'] & tc_final

    assert np.array_equal(et_final & tc_final, et_final), \
        'enforce_nesting invariant violated: ET_final is not a subset of TC_final'
    assert np.array_equal(tc_final & wt, tc_final), \
        'enforce_nesting invariant violated: TC_final is not a subset of WT'

    return {'WT': wt, 'TC': tc_final, 'ET': et_final}


def postprocess_regions(pred_masks, policy):
    """Orchestrates the full prediction post-processing pipeline for one
    volume's {'WT','TC','ET'} prediction masks:
      1. apply each region's configured method/min_component_voxels
      2. apply the ET volume threshold to the (already component-filtered) ET
      3. if policy['postproc'].get('enforce_nesting', True), re-nest the
         result (see enforce_nesting); otherwise return as-is.
    Ground truth masks must NEVER be passed through this function -- only
    predictions are post-processed; comparing against a post-processed
    ground truth would no longer measure prediction quality.
    """
    postproc_cfg = policy['postproc']
    out = {}
    for region in REGIONS:
        region_cfg = postproc_cfg[region]
        out[region] = apply_region_postproc(
            pred_masks[region], region_cfg['method'], region_cfg.get('min_component_voxels', 50))

    out['ET'] = apply_et_volume_threshold(out['ET'], postproc_cfg['et_volume_threshold'])

    if postproc_cfg.get('enforce_nesting', True):
        out = enforce_nesting(out)
    return out


# ---------------------------------------------------------------------------
# Per-region metrics
# ---------------------------------------------------------------------------

def binary_dice(pred_mask, gt_mask, eps=1e-8):
    """Standard Dice coefficient, float64 accumulation."""
    p = pred_mask.astype(np.float64)
    g = gt_mask.astype(np.float64)
    intersect = 2.0 * float(np.sum(p * g)) + eps
    denom = float(np.sum(p)) + float(np.sum(g)) + eps
    return intersect / denom


def sensitivity_specificity(pred_mask, gt_mask):
    """Returns (sensitivity, specificity) as floats, or None where undefined
    (no ground-truth positives for sensitivity, no ground-truth negatives for
    specificity)."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    tp = int(np.sum(pred_mask & gt_mask))
    fn = int(np.sum(~pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & ~gt_mask))
    tn = int(np.sum(~pred_mask & ~gt_mask))

    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else None
    return sensitivity, specificity


def mismatch_empty_penalty(shape, spacing_mm):
    """BraTS-style failure penalty for HD95 when exactly one of pred/gt is
    empty: the physical image diagonal."""
    return float(np.linalg.norm(np.asarray(shape, dtype=np.float64) * np.asarray(spacing_mm, dtype=np.float64)))


def hd95_from_masks(pred_mask, gt_mask, spacing_mm, penalty):
    """(value, tag): both empty -> (0.0, 'both_empty'); exactly one empty ->
    (penalty, 'mismatch_empty'); otherwise (medpy HD95, 'normal')."""
    from medpy.metric.binary import hd95 as medpy_hd95

    p_empty, g_empty = not pred_mask.any(), not gt_mask.any()
    if p_empty and g_empty:
        return 0.0, 'both_empty'
    if p_empty or g_empty:
        return penalty, 'mismatch_empty'
    return float(medpy_hd95(pred_mask, gt_mask, voxelspacing=spacing_mm)), 'normal'


def normalised_surface_distance(pred_mask, gt_mask, spacing_mm, tolerance_mm):
    """Normalised Surface Dice (NSD) at a given tolerance in mm.

    Edge-case convention (mirrors HD95's both_empty/mismatch_empty split, on
    NSD's natural [0,1] scale rather than a distance in mm): both masks empty
    -> 1.0 (perfect agreement that there's nothing there); exactly one empty
    -> 0.0 (total disagreement -- no surface distance is well defined).
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    p_empty, g_empty = not pred_mask.any(), not gt_mask.any()
    if p_empty and g_empty:
        return 1.0
    if p_empty or g_empty:
        return 0.0

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred_mask ^ ndimage.binary_erosion(pred_mask, structure=structure, border_value=0)
    gt_surface = gt_mask ^ ndimage.binary_erosion(gt_mask, structure=structure, border_value=0)

    spacing_mm = np.asarray(spacing_mm, dtype=np.float64)
    dist_to_gt = ndimage.distance_transform_edt(~gt_mask, sampling=spacing_mm)
    dist_to_pred = ndimage.distance_transform_edt(~pred_mask, sampling=spacing_mm)

    n_pred_surface = int(pred_surface.sum())
    n_gt_surface = int(gt_surface.sum())
    if n_pred_surface + n_gt_surface == 0:
        # Degenerate: masks are non-empty but have no detectable surface
        # (e.g. every foreground voxel is interior with no border, which in
        # practice only happens for a mask filling the whole volume).
        return 1.0

    within_pred = int(np.sum(dist_to_gt[pred_surface] <= tolerance_mm))
    within_gt = int(np.sum(dist_to_pred[gt_surface] <= tolerance_mm))
    return (within_pred + within_gt) / (n_pred_surface + n_gt_surface)


# ---------------------------------------------------------------------------
# The single entrypoint
# ---------------------------------------------------------------------------

def score(pred, gt, spacing_mm, policy):
    """The single scoring entrypoint used for EVERY entity (each personalised
    client, and the residual-zeroed/global model alike) -- there is no
    special-cased "global model" branch anywhere in this function. It only
    ever sees arrays, never knows what produced them.

    pred, gt: EITHER a 3D integer label-map ndarray (run through region_masks)
    OR an already-split dict {'WT','TC','ET'} of bool region masks (detected
    via isinstance(x, dict)), so a caller can re-score from previously-saved
    per-region .npz masks without re-deriving from a 4-class label map.

    Returns {region: {...} for region in REGIONS} with an identical key
    structure regardless of what pred/gt represent.
    """
    pred_masks = pred if isinstance(pred, dict) else region_masks(pred)
    gt_masks = gt if isinstance(gt, dict) else region_masks(gt)

    shape = pred_masks['WT'].shape
    penalty = mismatch_empty_penalty(shape, spacing_mm)
    tolerances = policy.get('nsd_tolerances_mm', [1.0, 3.0])

    # Nesting is cross-region, so post-process the whole pred_masks dict once.
    postproc_masks = postprocess_regions(pred_masks, policy)

    result = {}
    for region in REGIONS:
        p_raw = pred_masks[region]
        p_post = postproc_masks[region]
        g = gt_masks[region]

        dice_raw = binary_dice(p_raw, g)
        dice_postproc = binary_dice(p_post, g)

        hd95_raw, edge_raw_tag = hd95_from_masks(p_raw, g, spacing_mm, penalty)
        hd95_postproc, edge_postproc_tag = hd95_from_masks(p_post, g, spacing_mm, penalty)

        sensitivity_raw, specificity_raw = sensitivity_specificity(p_raw, g)
        sensitivity_postproc, specificity_postproc = sensitivity_specificity(p_post, g)

        nsd_raw = {str(tol): normalised_surface_distance(p_raw, g, spacing_mm, tol) for tol in tolerances}
        nsd_postproc = {str(tol): normalised_surface_distance(p_post, g, spacing_mm, tol) for tol in tolerances}

        result[region] = {
            'dice_raw': dice_raw,
            'dice_postproc': dice_postproc,
            'hd95_raw': hd95_raw,
            'hd95_postproc': hd95_postproc,
            'hd95_raw_valid_pairs_only': hd95_raw if edge_raw_tag == 'normal' else None,
            'hd95_postproc_valid_pairs_only': hd95_postproc if edge_postproc_tag == 'normal' else None,
            'sensitivity_raw': sensitivity_raw,
            'specificity_raw': specificity_raw,
            'sensitivity_postproc': sensitivity_postproc,
            'specificity_postproc': specificity_postproc,
            'nsd_raw': nsd_raw,
            'nsd_postproc': nsd_postproc,
            'edge_counts': {
                'both_empty': int(edge_raw_tag == 'both_empty'),
                'mismatch_empty': int(edge_raw_tag == 'mismatch_empty'),
                'normal': int(edge_raw_tag == 'normal'),
            },
        }
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    def make_blob(shape, center, radius):
        out = np.zeros(shape, dtype=bool)
        cz, cy, cx = center
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        out[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = True
        return out

    shape = (40, 40, 40)

    # --- region_masks -------------------------------------------------
    label_map = np.zeros(shape, dtype=np.int64)
    label_map[5:10, 5:10, 5:10] = 1  # NET
    label_map[10:15, 10:15, 10:15] = 2  # ED
    label_map[15:20, 15:20, 15:20] = 3  # ET
    rm = region_masks(label_map)
    assert set(rm.keys()) == {'WT', 'TC', 'ET'}
    assert rm['WT'].sum() == (label_map != 0).sum()
    assert rm['TC'].sum() == np.sum((label_map == 1) | (label_map == 3))
    assert rm['ET'].sum() == np.sum(label_map == 3)
    # nesting holds on the raw label map by construction
    assert np.array_equal(rm['ET'] & rm['TC'], rm['ET'])
    assert np.array_equal(rm['TC'] & rm['WT'], rm['TC'])
    print('region_masks: OK')

    # --- load_policy / DEFAULT_POLICY / validation ---------------------
    assert DEFAULT_POLICY['postproc']['WT']['method'] == 'largest_cc'
    assert DEFAULT_POLICY['postproc']['TC']['method'] == 'size_filter'
    assert DEFAULT_POLICY['postproc']['ET']['method'] == 'size_filter'
    _validate_policy(DEFAULT_POLICY)  # must not raise

    bad_policy = {
        'postproc': {
            'WT': {'method': 'not_a_real_method', 'min_component_voxels': 50},
            'TC': {'method': 'size_filter', 'min_component_voxels': 50},
            'ET': {'method': 'size_filter', 'min_component_voxels': 50},
            'et_volume_threshold': 500,
            'enforce_nesting': True,
        },
    }
    try:
        _validate_policy(bad_policy)
        raise AssertionError('expected ValueError for invalid postproc method')
    except ValueError:
        pass

    import os
    import tempfile
    import yaml as _yaml
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy_path = os.path.join(tmp_dir, 'metric_policy.yaml')
        with open(policy_path, 'w') as f:
            _yaml.safe_dump(DEFAULT_POLICY, f)
        loaded = load_policy(policy_path)
        assert loaded == DEFAULT_POLICY

        bad_path = os.path.join(tmp_dir, 'bad_policy.yaml')
        with open(bad_path, 'w') as f:
            _yaml.safe_dump(bad_policy, f)
        try:
            load_policy(bad_path)
            raise AssertionError('expected ValueError from load_policy on invalid file')
        except ValueError:
            pass
    print('load_policy: OK')

    # --- apply_region_postproc: largest_cc keeps only the big blob -----
    mask = make_blob(shape, (10, 10, 10), 6)  # big blob, ~900 voxels
    rng = np.random.RandomState(0)
    for _ in range(15):
        z, y, x = rng.randint(0, 40, size=3)
        if not mask[z, y, x]:
            mask[z, y, x] = True  # scattered isolated 1-voxel false positives
    big_blob_size = int(make_blob(shape, (10, 10, 10), 6).sum())
    kept = apply_region_postproc(mask, 'largest_cc')
    assert int(kept.sum()) == big_blob_size, 'largest_cc must keep exactly the big blob, dropping all specks'
    assert np.array_equal(kept, make_blob(shape, (10, 10, 10), 6))
    print('apply_region_postproc largest_cc: OK')

    # --- apply_region_postproc: size_filter keeps BOTH comparable blobs ---
    blob_a = make_blob(shape, (8, 8, 8), 4)
    blob_b = make_blob(shape, (30, 30, 30), 4)
    two_blob_mask = blob_a | blob_b
    size_a = int(blob_a.sum())
    size_b = int(blob_b.sum())
    assert size_a >= 50 and size_b >= 50, 'test blobs must clear the min_component_voxels threshold'
    filtered = apply_region_postproc(two_blob_mask, 'size_filter', min_component_voxels=50)
    assert np.array_equal(filtered, two_blob_mask), 'size_filter must keep BOTH qualifying components'
    largest_only = apply_region_postproc(two_blob_mask, 'largest_cc')
    assert int(largest_only.sum()) < int(filtered.sum()), \
        'largest_cc must collapse to one component -- size_filter must not'
    print('apply_region_postproc size_filter (multi-component): OK')

    # invalid method
    try:
        apply_region_postproc(mask, 'bogus')
        raise AssertionError('expected ValueError for invalid postproc method')
    except ValueError:
        pass

    # empty / <=1 component mask returned unchanged
    empty_mask = np.zeros(shape, dtype=bool)
    assert np.array_equal(apply_region_postproc(empty_mask, 'largest_cc'), empty_mask)
    single_blob = make_blob(shape, (10, 10, 10), 3)
    assert np.array_equal(apply_region_postproc(single_blob, 'largest_cc'), single_blob)
    print('apply_region_postproc edge cases: OK')

    # --- apply_et_volume_threshold --------------------------------------
    small_et = make_blob(shape, (10, 10, 10), 2)  # small, well under 500 voxels
    large_et = make_blob(shape, (20, 20, 20), 10)  # large, well over 500 voxels
    assert int(small_et.sum()) < 500
    assert int(large_et.sum()) >= 500
    thresholded_small = apply_et_volume_threshold(small_et, 500)
    thresholded_large = apply_et_volume_threshold(large_et, 500)
    assert not thresholded_small.any(), 'small ET below threshold must be zeroed'
    assert np.array_equal(thresholded_large, large_et), 'large ET above threshold must be untouched'
    # disabled (<=0) is a no-op even for the small mask
    assert np.array_equal(apply_et_volume_threshold(small_et, 0), small_et)
    assert np.array_equal(apply_et_volume_threshold(small_et, -1), small_et)
    print('apply_et_volume_threshold: OK')

    # --- enforce_nesting: TC has drifted outside WT ---------------------
    wt = make_blob(shape, (15, 15, 15), 8)
    tc = make_blob(shape, (15, 15, 15), 5).copy()
    drift_voxel = (35, 35, 35)
    assert not wt[drift_voxel]
    tc[drift_voxel] = True  # simulate independent-filter drift: TC voxel outside WT
    et = make_blob(shape, (15, 15, 15), 2).copy()
    nested = enforce_nesting({'WT': wt, 'TC': tc, 'ET': et})
    assert not nested['TC'][drift_voxel], 'drifted TC voxel outside WT must be removed'
    assert np.array_equal(nested['TC'], tc & wt)
    assert np.array_equal(nested['WT'], wt)
    assert np.array_equal(nested['ET'] & nested['TC'], nested['ET'])
    assert np.array_equal(nested['TC'] & nested['WT'], nested['TC'])
    print('enforce_nesting: OK')

    # --- postprocess_regions orchestration -------------------------------
    pred_masks_for_pp = {
        'WT': make_blob(shape, (15, 15, 15), 10),
        'TC': make_blob(shape, (15, 15, 15), 6),
        'ET': make_blob(shape, (15, 15, 15), 1),  # tiny ET, under the 500-voxel threshold
    }
    pp = postprocess_regions(pred_masks_for_pp, DEFAULT_POLICY)
    assert set(pp.keys()) == {'WT', 'TC', 'ET'}
    assert not pp['ET'].any(), 'tiny ET must be zeroed by et_volume_threshold'
    assert np.array_equal(pp['TC'] & pp['WT'], pp['TC'])
    print('postprocess_regions: OK')

    # --- binary_dice -------------------------------------------------
    a = make_blob(shape, (10, 10, 10), 5)
    assert abs(binary_dice(a, a) - 1.0) < 1e-9
    b_empty = np.zeros(shape, dtype=bool)
    assert binary_dice(b_empty, b_empty) > 0.99  # eps-only, ~1.0
    disjoint = make_blob(shape, (35, 35, 35), 2)
    assert binary_dice(a, disjoint) < 1e-6
    print('binary_dice: OK')

    # --- sensitivity_specificity --------------------------------------
    sens, spec = sensitivity_specificity(a, a)
    assert sens == 1.0 and spec == 1.0
    sens_none, spec_full = sensitivity_specificity(b_empty, b_empty)
    assert sens_none is None  # no GT positives -> undefined
    assert spec_full == 1.0
    sens_zero, spec_none = sensitivity_specificity(a, b_empty)
    assert sens_zero is None  # gt has no positives
    assert spec_none is not None and spec_none < 1.0  # a's voxels are false positives
    print('sensitivity_specificity: OK')

    # --- mismatch_empty_penalty / hd95_from_masks -----------------------
    spacing = (1.0, 1.0, 1.0)
    penalty = mismatch_empty_penalty(shape, spacing)
    expected_penalty = float(np.linalg.norm(np.array(shape, dtype=np.float64)))
    assert abs(penalty - expected_penalty) < 1e-9

    val, tag = hd95_from_masks(b_empty, b_empty, spacing, penalty)
    assert (val, tag) == (0.0, 'both_empty')
    val, tag = hd95_from_masks(a, b_empty, spacing, penalty)
    assert tag == 'mismatch_empty' and abs(val - penalty) < 1e-9
    val, tag = hd95_from_masks(b_empty, a, spacing, penalty)
    assert tag == 'mismatch_empty' and abs(val - penalty) < 1e-9
    val, tag = hd95_from_masks(a, a, spacing, penalty)
    assert tag == 'normal' and val == 0.0
    print('hd95_from_masks: OK')

    # --- normalised_surface_distance -------------------------------------
    nontrivial = make_blob(shape, (15, 15, 15), 7)
    nsd_identical = normalised_surface_distance(nontrivial, nontrivial, spacing, 1.0)
    assert abs(nsd_identical - 1.0) < 1e-9
    nsd_identical_big_tol = normalised_surface_distance(nontrivial, nontrivial, spacing, 5.0)
    assert abs(nsd_identical_big_tol - 1.0) < 1e-9

    far_a = make_blob(shape, (3, 3, 3), 2)
    far_b = make_blob(shape, (36, 36, 36), 2)
    nsd_far = normalised_surface_distance(far_a, far_b, spacing, 0.5)
    assert nsd_far == 0.0

    assert normalised_surface_distance(b_empty, b_empty, spacing, 1.0) == 1.0
    assert normalised_surface_distance(nontrivial, b_empty, spacing, 1.0) == 0.0
    assert normalised_surface_distance(b_empty, nontrivial, spacing, 1.0) == 0.0
    print('normalised_surface_distance: OK')

    # --- score(): identical key sets for two different "entities" ------
    def synthetic_label_map(seed):
        rng_local = np.random.RandomState(seed)
        lm = np.zeros(shape, dtype=np.int64)
        cz, cy, cx = 20, 20, 20
        lm[make_blob(shape, (cz, cy, cx), 9)] = 1
        lm[make_blob(shape, (cz, cy, cx), 6)] = 2
        lm[make_blob(shape, (cz, cy, cx), 3)] = 3
        # jitter with a few isolated speckle voxels to make the two predictions
        # genuinely different, simulating "a client" vs "the global model"
        for _ in range(10):
            z, y, x = rng_local.randint(0, 40, size=3)
            lm[z, y, x] = 1
        return lm

    gt_label = synthetic_label_map(seed=0)
    client_pred_label = synthetic_label_map(seed=1)
    global_pred_label = synthetic_label_map(seed=2)

    result_client = score(client_pred_label, gt_label, spacing, DEFAULT_POLICY)
    result_global = score(global_pred_label, gt_label, spacing, DEFAULT_POLICY)

    assert set(result_client.keys()) == set(result_global.keys()) == set(REGIONS)
    for region in REGIONS:
        assert set(result_client[region].keys()) == set(result_global[region].keys()), \
            'score() must return identical key structure regardless of which entity produced pred'
    print('score(): identical key sets for two different entities: OK')

    # --- score(): label map vs pre-split dict give consistent dice_raw ---
    pred_masks_dict = region_masks(client_pred_label)
    gt_masks_dict = region_masks(gt_label)
    result_from_labels = score(client_pred_label, gt_label, spacing, DEFAULT_POLICY)
    result_from_masks = score(pred_masks_dict, gt_masks_dict, spacing, DEFAULT_POLICY)
    for region in REGIONS:
        assert abs(result_from_labels[region]['dice_raw'] - result_from_masks[region]['dice_raw']) < 1e-12
        assert set(result_from_labels[region].keys()) == set(result_from_masks[region].keys())
    print('score(): label-map input vs pre-split dict input agree: OK')

    # --- score(): full sanity on structure/values for one region --------
    et_entry = result_from_labels['ET']
    for key in ('dice_raw', 'dice_postproc', 'hd95_raw', 'hd95_postproc',
                'hd95_raw_valid_pairs_only', 'hd95_postproc_valid_pairs_only',
                'sensitivity_raw', 'specificity_raw', 'sensitivity_postproc', 'specificity_postproc',
                'nsd_raw', 'nsd_postproc', 'edge_counts'):
        assert key in et_entry, 'missing key {!r} in score() output'.format(key)
    assert set(et_entry['nsd_raw'].keys()) == {str(t) for t in DEFAULT_POLICY['nsd_tolerances_mm']}
    assert sum(et_entry['edge_counts'].values()) == 1
    print('score(): output structure sanity: OK')

    print('ALL TESTS PASSED')

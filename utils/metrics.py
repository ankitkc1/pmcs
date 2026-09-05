"""
Additional, opt-in evaluation metrics for PMCS federated training: HD95,
measured (not estimated) communication cost, rounds-to-target-Dice, the
private/shared parameter split of the fusion adapter, personalisation gain,
client-Dice variation, and global-model construction/evaluation.

Nothing here changes training, aggregation, or data loading. The only model
mutation performed anywhere in this module is on throwaway the global model built from already-aggregated global
state
"""
import copy
import csv
import json
import os

import numpy as np
import torch

from utils.predict import softmax_output_dice_class4

REGIONS = ('WT', 'TC', 'ET')
HD95_BOTH_EMPTY = 0.0
HD95_MISMATCH_EMPTY = 373.13


# HD95

def _region_masks(label_map):
    """label_map: numpy array of ints in {0,1,2,3} (NET=1, ED=2, ET=3, BG=0),
    matching the label convention already used by softmax_output_dice_class4."""
    net = label_map == 1
    ed = label_map == 2
    et = label_map == 3
    return {
        'WT': net | ed | et,
        'TC': net | et,
        'ET': et,
    }


def hd95_case(pred_label, gt_label, spacing=(1.0, 1.0, 1.0)):
    """
    Per-case, per-region 95th-percentile Hausdorff distance in mm, using the
    BraTS edge-case convention: both empty -> 0, exactly one empty -> 373.13.
    Returns (values, tags) dicts keyed by region name.
    """
    from medpy.metric.binary import hd95 as medpy_hd95

    pred_masks = _region_masks(pred_label)
    gt_masks = _region_masks(gt_label)

    values, tags = {}, {}
    for region in REGIONS:
        p, g = pred_masks[region], gt_masks[region]
        p_empty, g_empty = not p.any(), not g.any()
        if p_empty and g_empty:
            values[region] = HD95_BOTH_EMPTY
            tags[region] = 'both_empty'
        elif p_empty or g_empty:
            values[region] = HD95_MISMATCH_EMPTY
            tags[region] = 'mismatch_empty'
        else:
            values[region] = float(medpy_hd95(p, g, voxelspacing=spacing))
            tags[region] = 'normal'
    return values, tags


# Sliding-window evaluator with optional HD95 on top.

def evaluate_client(args, test_loader, model, device, modal_mask, compute_hd95=False, spacing=(1.0, 1.0, 1.0)):
    """
    Sliding-window inference identical to utils.predict.local_test (same
    patch_size=80, same overlap/weighting), reusing softmax_output_dice_class4
    for Dice so results agree with the existing eval path.
    """
    model = model.to(device)
    H, W, T = 240, 240, 155
    patch_size = 80
    model.eval()
    one_tensor = torch.ones(1, patch_size, patch_size, 80).float().to(device)

    dice_sum = np.zeros(3, dtype=np.float64)
    hd95_sum = np.zeros(3, dtype=np.float64)
    edge_counts = {r: {'both_empty': 0, 'mismatch_empty': 0, 'normal': 0} for r in REGIONS}
    n_cases = 0

    mask_t = torch.from_numpy(np.array(modal_mask)).unsqueeze(0).to(device)

    with torch.no_grad():
        for data in test_loader:
            target = data[1].to(device)
            x = data[0].to(device)
            names = data[-1]
            _, _, Hh, Ww, Zz = x.size()

            h_cnt = int(np.ceil((Hh - patch_size) / (patch_size * 0.5)))
            h_idx_list = [h * int(patch_size * 0.5) for h in range(h_cnt)] + [Hh - patch_size]
            w_cnt = int(np.ceil((Ww - patch_size) / (patch_size * 0.5)))
            w_idx_list = [w * int(patch_size * 0.5) for w in range(w_cnt)] + [Ww - patch_size]
            z_cnt = int(np.ceil((Zz - 80) / (80 * 0.5)))
            z_idx_list = [z * int(80 * 0.5) for z in range(z_cnt)] + [Zz - 80]

            weight1 = torch.zeros(1, 1, Hh, Ww, Zz).float().to(device)
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h + patch_size, w:w + patch_size, z:z + 80] += one_tensor
            weight = weight1.repeat(len(names), 4, 1, 1, 1)

            pred = torch.zeros(len(names), 4, Hh, Ww, Zz).float().to(device)
            model.is_training = False
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = x[:, :, h:h + patch_size, w:w + patch_size, z:z + 80]
                        pred_part, _, _ = model(x_input, mask_t, None, None, None, None)
                        pred[:, :, h:h + patch_size, w:w + patch_size, z:z + 80] += pred_part
            pred = pred / weight
            pred = pred[:, :, :H, :W, :T]
            pred = torch.argmax(pred, dim=1)

            _, scores_evaluation = softmax_output_dice_class4(pred, target)

            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            for k in range(len(names)):
                dice_sum += scores_evaluation[k, :3]
                n_cases += 1
                if compute_hd95:
                    values, tags = hd95_case(pred_np[k], target_np[k], spacing=spacing)
                    for ri, region in enumerate(REGIONS):
                        hd95_sum[ri] += values[region]
                        edge_counts[region][tags[region]] += 1

    n_cases = max(n_cases, 1)
    result = {'dice': dice_sum / n_cases, 'n_cases': n_cases}
    if compute_hd95:
        result['hd95'] = hd95_sum / n_cases
        result['edge_counts'] = edge_counts
    return result


def zero_residual_copy(model):
    """Deep-copies `model` and zeroes every FusionAdapter residual (R_k) on
    the copy only. """
    model_copy = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in model_copy.named_parameters():
            if name.endswith('_residual'):
                p.zero_()
    return model_copy


def run_pooled(args, client_indices, task_fn, build_args_fn):
    """
    Generic multi-GPU pooled dispatcher, mirroring the branch/pool pattern
    already used for local_training/local_test in train_federated.py.
    """
    results = {}
    branch_num = len(client_indices) // args.num_devices
    if len(client_indices) % args.num_devices:
        branch_num += 1
    for branch in range(branch_num):
        ctx = torch.multiprocessing.get_context('spawn')
        pool = ctx.Pool(args.num_devices)
        pending = []
        for slot in range(args.num_devices):
            idx = slot + branch * args.num_devices
            if idx >= len(client_indices):
                break
            c = client_indices[idx]
            pending.append((c, pool.apply_async(task_fn, build_args_fn(c))))
        pool.close()
        pool.join()
        for c, r in pending:
            results[c] = r.get()
    return results


# Communication cost -- measured from the real tensors at the point they cross the client/server boundary

class CommTracker:
    def __init__(self, client_num):
        self.client_num = client_num
        self.uploaded = {c: {} for c in range(client_num)}
        self.downloaded = {c: {} for c in range(client_num)}

    def record_upload(self, client_idx, round_idx, encoders, decoder_state, mask):
        """
        Protocol-level upload: only encoders for modalities this client's mask
        holds , plus the decoder's shared prior (A0) -- never the private residual (R_k).
        """
        numel = 0
        for m in range(4):
            if bool(mask[m]):
                numel += sum(t.numel() for t in encoders[m].values())
        counted_keys = []
        for k, t in decoder_state.items():
            if k.endswith('_residual'):
                continue
            counted_keys.append(k)
            numel += t.numel()
        assert not any(k.endswith('_residual') for k in counted_keys), \
            'R_k (fusion adapter residual) must never be counted as uploaded'
        self.uploaded[client_idx][round_idx] = int(numel)
        return int(numel)

    def record_download(self, client_idx, round_idx, global_encoders, global_decoder_prior):
        """What broadcast_weights actually writes into this client: all 4
        global encoders + the decoder
        prior. Never the residual -- global_decoder_prior has no such keys."""
        numel = sum(t.numel() for enc in global_encoders for t in enc.values())
        counted_keys = list(global_decoder_prior.keys())
        assert not any(k.endswith('_residual') for k in counted_keys), \
            'R_k (fusion adapter residual) must never be counted as downloaded'
        numel += sum(t.numel() for t in global_decoder_prior.values())
        self.downloaded[client_idx][round_idx] = int(numel)
        return int(numel)

    def round_totals(self, round_idx):
        return {
            c: {
                'uploaded': self.uploaded[c].get(round_idx, 0),
                'downloaded': self.downloaded[c].get(round_idx, 0),
            }
            for c in range(self.client_num)
        }

    def summary(self):
        per_client = {}
        grand_uploaded = 0
        grand_downloaded = 0
        for c in range(self.client_num):
            up = sum(self.uploaded[c].values())
            down = sum(self.downloaded[c].values())
            grand_uploaded += up
            grand_downloaded += down
            per_client[c] = {
                'uploaded_params_total': up,
                'downloaded_params_total': down,
                'uploaded_bytes_total': up * 4,
                'downloaded_bytes_total': down * 4,
            }
        total_bytes = 4 * (grand_uploaded + grand_downloaded)
        return {
            'per_client': per_client,
            'uploaded_params_by_round': {c: dict(self.uploaded[c]) for c in range(self.client_num)},
            'downloaded_params_by_round': {c: dict(self.downloaded[c]) for c in range(self.client_num)},
            'total_bytes_all_clients_all_rounds': total_bytes,
        }

    def state_dict(self):
        return {'uploaded': self.uploaded, 'downloaded': self.downloaded}

    def load_state_dict(self, state):
        self.uploaded = {int(k): {int(rk): rv for rk, rv in v.items()} for k, v in state['uploaded'].items()}
        self.downloaded = {int(k): {int(rk): rv for rk, rv in v.items()} for k, v in state['downloaded'].items()}


# Private / shared parameter accounting for the fusion adapter

def compute_private_fraction(template_model, mask):

    encoder_modules = [
        template_model.flair_encoder, template_model.t1ce_encoder,
        template_model.t1_encoder, template_model.t2_encoder,
    ]
    held_encoders_numel = sum(
        p.numel() for m, enc in zip(mask, encoder_modules) if m for p in enc.parameters())
    unheld_encoders_numel = sum(
        p.numel() for m, enc in zip(mask, encoder_modules) if not m for p in enc.parameters())

    adapter = template_model.fusion_decoder.adapter
    private_numel = sum(p.numel() for n, p in adapter.named_parameters() if n.endswith('_residual'))
    a0_numel = sum(p.numel() for n, p in adapter.named_parameters() if n.endswith('_prior'))
    fusion_decoder_total = sum(p.numel() for p in template_model.fusion_decoder.parameters())
    rest_of_decoder_numel = fusion_decoder_total - private_numel - a0_numel
    modality_decoder_numel = sum(p.numel() for p in template_model.modality_decoder.parameters())

    shared_numel = held_encoders_numel + a0_numel + rest_of_decoder_numel
    total_model_denom = private_numel + shared_numel
    fusion_module_denom = private_numel + a0_numel

    return {
        'private_numel': int(private_numel),
        'shared_numel': int(shared_numel),
        'fraction_of_total_model': (private_numel / total_model_denom) if total_model_denom else 0.0,
        'fraction_of_fusion_module': (private_numel / fusion_module_denom) if fusion_module_denom else 0.0,
        'held_encoders_numel': int(held_encoders_numel),
        'unheld_encoders_numel': int(unheld_encoders_numel),
        'a0_numel': int(a0_numel),
        'rest_of_decoder_numel': int(rest_of_decoder_numel),
        'modality_decoder_numel_not_in_fraction': int(modality_decoder_numel),
    }


# Client-Dice variation

def client_dice_variation(dice_matrix):
    """dice_matrix: array-like [n_clients, 3] of (WT,TC,ET) dice in [0,1]."""
    dice_matrix = np.asarray(dice_matrix, dtype=np.float64)
    mean_per_client = dice_matrix.mean(axis=1)
    best = int(np.argmax(mean_per_client))
    worst = int(np.argmin(mean_per_client))
    return {
        'mean_dice_std': float(mean_per_client.std()),
        'mean_dice_range': float(mean_per_client.max() - mean_per_client.min()),
        'region_std': {r: float(dice_matrix[:, i].std()) for i, r in enumerate(REGIONS)},
        'best_client': best,
        'worst_client': worst,
        'best_client_mean_dice': float(mean_per_client[best]),
        'worst_client_mean_dice': float(mean_per_client[worst]),
    }


# Rounds-to-target-Dice

class TargetDiceTracker:
    def __init__(self, targets, client_num):
        self.targets = [float(t) for t in targets]
        self.client_num = client_num
        self.mean_hit_round = {self._key(t): None for t in self.targets}
        self.per_client_hit_round = {c: {self._key(t): None for t in self.targets} for c in range(client_num)}
        self.per_region_hit_round = {r: {self._key(t): None for t in self.targets} for r in REGIONS}

    @staticmethod
    def _key(t):
        return '{:g}'.format(t)

    def update(self, round_idx, dice_matrix):
        """dice_matrix: [n_clients, 3] Dice in [0,1]; targets are percentages."""
        dice_matrix = np.asarray(dice_matrix, dtype=np.float64) * 100.0
        mean_per_client = dice_matrix.mean(axis=1)
        overall_mean = float(mean_per_client.mean())
        region_mean = dice_matrix.mean(axis=0)

        for t in self.targets:
            k = self._key(t)
            if self.mean_hit_round[k] is None and overall_mean >= t:
                self.mean_hit_round[k] = round_idx
            for c in range(self.client_num):
                if self.per_client_hit_round[c][k] is None and mean_per_client[c] >= t:
                    self.per_client_hit_round[c][k] = round_idx
            for i, r in enumerate(REGIONS):
                if self.per_region_hit_round[r][k] is None and region_mean[i] >= t:
                    self.per_region_hit_round[r][k] = round_idx


# ---------------------------------------------------------------------------
# metrics.json / metrics.csv persistence
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, torch.device):
        return str(o)
    raise TypeError('object of type {} is not JSON serialisable'.format(type(o)))


class MetricsRecorder:
    def __init__(self, save_path, client_num):
        self.save_path = save_path
        self.json_path = os.path.join(save_path, 'metrics.json')
        self.csv_path = os.path.join(save_path, 'metrics.csv')
        self.client_num = client_num
        self.comm = CommTracker(client_num)
        self.rounds = {}
        self.static = {}
        self.target_tracker = None

    def init_target_tracker(self, targets):
        self.target_tracker = TargetDiceTracker(targets, self.client_num)

    def try_resume(self):
        if not os.path.exists(self.json_path):
            return
        with open(self.json_path) as f:
            data = json.load(f)
        self.static = data.get('static', self.static)
        self.rounds = {int(k): v for k, v in data.get('rounds', {}).items()}
        if 'comm_state' in data:
            self.comm.load_state_dict(data['comm_state'])
        saved_targets = data.get('rounds_to_target_dice')
        if self.target_tracker is not None and saved_targets:
            self.target_tracker.targets = saved_targets['targets']
            self.target_tracker.mean_hit_round = saved_targets['mean']
            self.target_tracker.per_client_hit_round = {
                int(k): v for k, v in saved_targets['per_client'].items()}
            self.target_tracker.per_region_hit_round = saved_targets['per_region']

    def set_static(self, key, value):
        self.static[key] = value

    def record_round(self, round_idx, payload):
        self.rounds.setdefault(round_idx, {}).update(payload)
        if self.target_tracker is not None and 'dice_matrix' in payload:
            self.target_tracker.update(round_idx, payload['dice_matrix'])

    def _rounds_to_target_summary(self):
        if self.target_tracker is None:
            return None
        return {
            'targets': self.target_tracker.targets,
            'mean': self.target_tracker.mean_hit_round,
            'per_client': self.target_tracker.per_client_hit_round,
            'per_region': self.target_tracker.per_region_hit_round,
        }

    def flush(self):
        data = {
            'static': self.static,
            'rounds': self.rounds,
            'communication': self.comm.summary(),
            'comm_state': self.comm.state_dict(),
            'rounds_to_target_dice': self._rounds_to_target_summary(),
        }
        tmp_path = self.json_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2, default=_json_default)
        os.replace(tmp_path, self.json_path)
        self._flush_csv()

    def _flush_csv(self):
        rows = []
        for round_idx in sorted(self.rounds.keys()):
            payload = self.rounds[round_idx]
            dice_matrix = payload.get('dice_matrix')
            hd95_matrix = payload.get('hd95_matrix')
            gain_matrix = payload.get('personalisation_gain_matrix')
            test_dice_matrix = payload.get('test_dice_matrix')
            test_hd95_matrix = payload.get('test_hd95_matrix')
            test_gain_matrix = payload.get('test_personalisation_gain_matrix')
            comm_per_client = payload.get('comm_per_client', {})
            global_minus_client = payload.get('global_minus_client', {})
            test_global_minus_client = payload.get('test_global_minus_client', {})
            for c in range(self.client_num):
                row = {'round': round_idx, 'client': c + 1}
                if dice_matrix is not None:
                    for i, r in enumerate(REGIONS):
                        row['dice_' + r] = dice_matrix[c][i]
                if hd95_matrix is not None:
                    for i, r in enumerate(REGIONS):
                        row['hd95_' + r] = hd95_matrix[c][i]
                if gain_matrix is not None:
                    for i, r in enumerate(REGIONS):
                        row['pers_gain_' + r] = gain_matrix[c][i]
                if test_dice_matrix is not None:
                    for i, r in enumerate(REGIONS):
                        row['test_dice_' + r] = test_dice_matrix[c][i]
                if test_hd95_matrix is not None:
                    for i, r in enumerate(REGIONS):
                        row['test_hd95_' + r] = test_hd95_matrix[c][i]
                if test_gain_matrix is not None:
                    for i, r in enumerate(REGIONS):
                        row['test_pers_gain_' + r] = test_gain_matrix[c][i]
                comm = comm_per_client.get(c) or comm_per_client.get(str(c))
                if comm is not None:
                    row['uploaded_params'] = comm['uploaded']
                    row['downloaded_params'] = comm['downloaded']
                gmc = global_minus_client.get(c) or global_minus_client.get(str(c))
                if gmc is not None:
                    for i, r in enumerate(REGIONS):
                        row['global_minus_client_' + r] = gmc[i]
                test_gmc = test_global_minus_client.get(c) or test_global_minus_client.get(str(c))
                if test_gmc is not None:
                    for i, r in enumerate(REGIONS):
                        row['test_global_minus_client_' + r] = test_gmc[i]
                rows.append(row)
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row.keys()},
                             key=lambda k: (k != 'round', k != 'client', k))
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

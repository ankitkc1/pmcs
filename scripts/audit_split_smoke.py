#!/usr/bin/env python3
"""Audit paired two-round Split A/B smoke-test outputs."""

import argparse
import json
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_result(path, expected_split):
    path = Path(path)
    with (path / 'metrics.json').open('r') as handle:
        metrics = json.load(handle)

    split = metrics['static']['split']
    require(split['split_id'] == expected_split, 'wrong split id in metrics')
    require(split['data_seed'] == 20260905, 'wrong data_seed in metrics')

    rounds = metrics['rounds']
    require(sorted(rounds, key=int) == ['0', '1'], 'smoke test must contain rounds 0 and 1')
    require('test_dice_matrix' not in rounds['0'], 'test data was accessed before final round')
    require('global_model' not in rounds['0'], 'global test was accessed before final round')

    final = rounds['1']
    require(len(final['validation_dice_matrix']) == 8, 'validation matrix must contain 8 clients')
    require(len(final['test_dice_matrix']) == 8, 'test matrix must contain 8 clients')
    require(all(len(row) == 3 for row in final['test_dice_matrix']), 'test Dice rows must be WT/TC/ET')
    require(final['global_model']['n_cases'] == 50, 'global test must contain 50 cases')
    require(final['global_model']['partition'] == 'global_held_out_test', 'wrong global partition label')
    require(final['global_model']['private_residual_zeroed'] is True,
            'global model private residual was not declared zeroed')

    checkpoint = torch.load(path / 'model_files' / 'last.pth', map_location='cpu')
    require(checkpoint['round'] == 2, 'two-round smoke checkpoint must store round 2')
    require(checkpoint['split_metadata']['split_id'] == expected_split,
            'checkpoint split id mismatch')
    require(checkpoint['split_metadata']['patient_assignment_fingerprint'] ==
            split['patient_assignment_fingerprint'], 'checkpoint/metrics assignment mismatch')
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split_a_result', type=Path)
    parser.add_argument('split_b_result', type=Path)
    args = parser.parse_args()

    split_a = load_result(args.split_a_result, 'A')
    split_b = load_result(args.split_b_result, 'B')
    require(split_a['mapping_fingerprint'] == split_b['mapping_fingerprint'],
            'A/B mapping fingerprints differ')
    require(split_a['patient_assignment_fingerprint'] ==
            split_b['patient_assignment_fingerprint'], 'A/B patient assignments differ')

    for mask_a, mask_b in zip(split_a['masks'], split_b['masks']):
        require(mask_b == [mask_a[1], mask_a[0], mask_a[2], mask_a[3]],
                'A/B masks are not a pure FLAIR/T1ce swap')

    print('PAIRED SPLIT SMOKE AUDIT PASSED')
    print('Assignment fingerprint: {}'.format(split_a['patient_assignment_fingerprint']))
    print('Both checkpoints completed exactly 2 rounds')
    print('Test sets were reported only at the final round')


if __name__ == '__main__':
    main()

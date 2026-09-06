#!/usr/bin/env python3
"""Audit a completed PMCS materialized-split training run."""

import argparse
import json
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--split', choices=('A', 'B'), required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--eval_every', type=int, required=True)
    args = parser.parse_args()

    with (args.result_dir / 'metrics.json').open('r') as handle:
        metrics = json.load(handle)
    recorded = {int(key): value for key, value in metrics['rounds'].items()}
    expected = list(range(args.eval_every - 1, args.rounds, args.eval_every))
    if not expected or expected[-1] != args.rounds - 1:
        expected.append(args.rounds - 1)
    require(sorted(recorded) == expected, 'unexpected evaluation rounds')

    for round_index in expected[:-1]:
        require('test_dice_matrix' not in recorded[round_index],
                'client test accessed before final round')
        require('global_model' not in recorded[round_index],
                'global test accessed before final round')

    final = recorded[args.rounds - 1]
    require(len(final['validation_dice_matrix']) == 8, 'expected 8 validation clients')
    require(len(final['test_dice_matrix']) == 8, 'expected 8 test clients')
    require(len(final['test_hd95_matrix']) == 8, 'expected 8 client HD95 rows')
    require(final['hd95_policy']['mismatch_empty'] == 'physical_image_diagonal',
            'unexpected HD95 policy')
    require(final['global_model']['n_cases'] == 50, 'expected 50 global-test cases')
    require(final['global_model']['private_residual_zeroed'] is True,
            'global private residual was not zeroed')

    split = metrics['static']['split']
    require(split['split_id'] == args.split, 'metrics split mismatch')
    require(split['data_seed'] == 20260905, 'metrics data_seed mismatch')

    checkpoint = torch.load(
        args.result_dir / 'model_files' / 'last.pth', map_location='cpu')
    require(checkpoint['round'] == args.rounds, 'checkpoint round mismatch')
    require(checkpoint['split_metadata']['split_id'] == args.split,
            'checkpoint split mismatch')
    require(checkpoint['split_metadata']['patient_assignment_fingerprint'] ==
            split['patient_assignment_fingerprint'],
            'checkpoint/metrics assignment mismatch')

    print('FULL {}-ROUND AUDIT PASSED'.format(args.rounds))
    print('Split: {}'.format(args.split))
    print('Checkpoint round: {}'.format(checkpoint['round']))
    print('HD95 policy: physical image diagonal for one-empty cases')
    print('Result directory: {}'.format(args.result_dir))


if __name__ == '__main__':
    main()

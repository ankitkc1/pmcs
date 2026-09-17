# -*- coding: utf-8 -*-
"""Turn the exp2/*.json logs into the tables
"""
from __future__ import annotations

import argparse
import glob
import json
import os

MODS = ['FLAIR', 'T1ce', 'T1', 'T2']

NICE = {
    'uniform': 'uniform random',
    'round_robin': 'round robin',
    'powd_d_mult2': 'Power-of-Choice  d=2K',
    'powd_d_mult10': 'Power-of-Choice  d=N',
    'rpowd_d_mult2': 'rpow-d  (no probe)',
    'mfedmc_gamma1_delta0.2': 'MFedMC  gamma=1',
    'mfedmc_gamma2_delta0.2': 'MFedMC  gamma=2',
    'mfedmc_gamma4_delta0.2': 'MFedMC  gamma=4  (no upload filter)',
}
ORDER = list(NICE)


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, '*.json'))):
        tag = os.path.splitext(os.path.basename(p))[0]
        with open(p) as f:
            out[tag] = json.load(f)
    return out


def table(runs):
    keys = [k for k in ORDER if k in runs] + \
           [k for k in runs if k not in ORDER]
    print('=' * 108)
    print('TABLE 1  encoder coverage, staleness, fairness, communication')
    print('=' * 108)
    print(f'{"method":36}' + ''.join(f'{m:>8}' for m in MODS)
          + f'{"worst":>7}{"stall":>7}{"eJain":>7}{"cJain":>7}'
            f'{"wait":>6}{"contact":>8}{"probes":>8}{"GB up":>8}{"GB dn":>8}')
    for k in keys:
        s = runs[k]['summary']
        print(f'{NICE.get(k, k):36}'
              + ''.join(f'{v:8d}' for v in s['encoder_updates'])
              + f'{s["worst_encoder"]:7d}{s["max_encoder_stall"]:7d}'
                f'{s["encoder_jain"]:7.3f}{s["client_jain"]:7.3f}'
                f'{s["longest_client_wait"]:6d}{s["contacted_per_round"]:8.1f}'
                f'{s["probe_passes_total"]:8d}{s["GB_up"]:8.1f}{s["GB_down"]:8.1f}')

    print()
    print('=' * 108)
    print('TABLE 2  held but NOT aggregated  -- the cost of upload filtering')
    print('=' * 108)
    print(f'{"method":36}' + ''.join(f'{m:>10}' for m in MODS)
          + '   (rounds)')
    for k in keys:
        s = runs[k]['summary']
        print(f'{NICE.get(k, k):36}'
              + ''.join(f'{g:10d}' for g in s['held_minus_updated']))
    print('\nNon-zero only for methods with a modality-upload filter. No')
    print('published evaluation of those methods reports this column.')
    return keys


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='exp2')
    a = ap.parse_args()
    runs = load(a.dir)
    if not runs:
        raise SystemExit(f'no json found in {a.dir}/')
    print(f'loaded {len(runs)} runs from {a.dir}/\n')
    table(runs)

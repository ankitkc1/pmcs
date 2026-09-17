# -*- coding: utf-8 -*-
"""
Every selector here decides who trains and what they upload. Only pow-d and
MFedMC need a loss value to do it. In --dry mode those losses are supplied by a
stand-in model fitted to your K=8 results, so the whole experiment runs on CPU
in minutes.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from fl_selectors import CoverageLog, make_selector

MODS = ['FLAIR', 'T1ce', 'T1', 'T2']
REG = ['WT', 'TC', 'ET']

MASK_A = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0],
                   [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
MASK_B = MASK_A[:, [1, 0, 2, 3]]

# measured per-client macro Dice, shared 50-case test, K = 8, seed 42
MACRO = {'A': np.array([71.874, 72.260, 57.334, 54.253,
                        55.382, 48.943, 55.573, 43.583]),
         'B': np.array([75.216, 73.642, 74.877, 63.698,
                        69.401, 50.277, 63.683, 43.864])}

# measured payload model: 1.524 GB decoder + 0.880 GB per encoder, over 150 rounds
DEC_B = 1.524 * 2**30 / 150
ENC_B = np.full(4, 0.880 * 2**30 / 150)

# per-region modality impact, used only as a Shapley stand-in in --dry
IMPACT = np.array([[14.88, 12.60, 7.92], [-0.45, 13.98, 35.30],
                   [4.25, 5.59, 4.05], [2.40, 2.29, 3.48]])

T = 150
SEED = 42


# ============================================================ dry-run context
class DryContext:
    """Stand-in for the training loop. Loss decays toward a client-specific
    floor set by the measured K=8 score, so the ranking is realistic and moves
    a little with t rather than being frozen."""

    def __init__(self, macro, mask, rng):
        self.macro = np.asarray(macro, float)
        self.mask = mask
        self.rng = rng
        self.n_samples = {k: 50 for k in range(mask.shape[0])}
        self.encoder_bytes = {m: float(ENC_B[m]) for m in range(mask.shape[1])}
        self._t = 0

    def set_round(self, t):
        self._t = t

    def local_loss(self, ids):
        floor = (100.0 - self.macro) / 100.0
        decay = 0.6 * np.exp(-self._t / 40.0)
        noise = self.rng.normal(0, 0.01, size=len(self.macro))
        cur = floor + decay + noise
        return {int(k): float(cur[int(k)]) for k in ids}

    def shapley(self, k):
        held = [m for m in range(self.mask.shape[1]) if self.mask[k, m]]
        return {m: float(IMPACT[m].sum()) for m in held}


class StrictContext:
    """Raises on any attribute access. Proves a selector needs nothing."""
    def __getattr__(self, name):
        raise AssertionError(f'selector touched ctx.{name}')


# ============================================================ the run
CONFIGS = [
    ('uniform',      'uniform random',                    {}),
    ('round_robin',  'round robin',                       {}),
    ('powd',         'Power-of-Choice  d=2K',             dict(d_mult=2)),
    ('powd',         'Power-of-Choice  d=N (10K capped)',  dict(d_mult=10)),
    ('rpowd',        'rpow-d  (no probe, stale loss)',    dict(d_mult=2)),
    ('mfedmc',       'MFedMC  gamma=1 delta=0.2',         dict(gamma=1, delta=0.2)),
    ('mfedmc',       'MFedMC  gamma=2 delta=0.2',         dict(gamma=2, delta=0.2)),
    ('mfedmc',       'MFedMC  gamma=4 (no upload filter)', dict(gamma=4, delta=0.2)),
]


def run_one(name, kw, mask, macro, K, dry=True, ctx_factory=None, outdir='exp2'):
    rng = np.random.default_rng(SEED)
    sel = make_selector(name, mask, K, seed=SEED, **kw)
    log = CoverageLog(mask, DEC_B, ENC_B)
    ctx = (DryContext(macro, mask, rng) if dry else ctx_factory())
    for t in range(T):
        if hasattr(ctx, 'set_round'):
            ctx.set_round(t)
        plan = sel.plan_round(t, ctx)
        if not dry:
            # ---- the only lines that touch your training code ----
            #   train_clients(plan.train)
            #   aggregate(plan.upload, plan.aggregate_from)
            #   losses = evaluate_local_loss(plan.train)
            raise NotImplementedError('wire integrate() to your loop')
        losses = ctx.local_loss(plan.train)
        sel.observe(t, plan, losses_after=losses)
        log.record(t, plan)
    os.makedirs(outdir, exist_ok=True)
    tag = f'{name}_' + '_'.join(f'{k}{v}' for k, v in kw.items()) if kw else name
    log.save(os.path.join(outdir, f'{tag}.json'))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', default=True)
    ap.add_argument('--scenario', default='A', choices=['A', 'B'])
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--outdir', default='exp2')
    a = ap.parse_args()

    mask = MASK_A if a.scenario == 'A' else MASK_B
    macro = MACRO[a.scenario]
    scarce = int(mask.sum(0).argmin())

    print(f'EXPERIMENT 2  |  scenario {a.scenario}  |  K = {a.K}  |  T = {T}  '
          f'|  seed {SEED}')
    print(f'pools ' + '  '.join(f'{MODS[m]} {mask.sum(0)[m]}' for m in range(4))
          + f'   scarce = {MODS[scarce]}')
    print(f'{"DRY RUN - selection only, stand-in losses" if a.dry else "LIVE"}\n')

    rows = []
    for name, label, kw in CONFIGS:
        log = run_one(name, kw, mask, macro, a.K, dry=a.dry, outdir=a.outdir)
        s = log.summary()
        rows.append((label, s))
        print(f'--- {label} ---')
        log.report(MODS)
        print()

    print('=' * 104)
    print('SUMMARY TABLE  (this is the table for the supervisor)')
    print('=' * 104)
    hdr = (f'{"method":34}' + ''.join(f'{m:>8}' for m in MODS)
           + f'{"worst":>7}{"stall":>7}{"cJain":>7}{"wait":>6}'
             f'{"contact":>8}{"probes":>8}{"GB":>7}')
    print(hdr)
    for label, s in rows:
        print(f'{label:34}' + ''.join(f'{v:8d}' for v in s['encoder_updates'])
              + f'{s["worst_encoder"]:7d}{s["max_encoder_stall"]:7d}'
                f'{s["client_jain"]:7.3f}{s["longest_client_wait"]:6d}'
                f'{s["contacted_per_round"]:8.1f}{s["probe_passes_total"]:8d}'
                f'{s["GB_total"]:7.1f}')

    print(f'\n{"=" * 104}')
    print('HELD vs AGGREGATED  — the gap upload filtering costs')
    print('=' * 104)
    print(f'{"method":34}' + ''.join(f'{m:>10}' for m in MODS))
    for label, s in rows:
        print(f'{label:34}'
              + ''.join(f'{g:10d}' for g in s['held_minus_updated']))
    print('\nA positive entry is a round in which an aggregating client HELD that')
    print('encoder and did not upload it. Only MFedMC can be non-zero here, and')
    print('no published evaluation reports this column.')

    print(f'\nwrote per-round JSON to {a.outdir}/')


# ============================================================ integration
def integrate():
    def integrate():
    """Placeholder for future integration with the real training loop."""
    raise NotImplementedError(
        "Live training integration has not been implemented. "
        "Use the CPU dry-run simulator for now."
    )


if __name__ == '__main__':
    main()

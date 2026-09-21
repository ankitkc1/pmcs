# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import json
import sys

import numpy as np

from fl_selectors import CoverageLog, make_selector

MODS = ['FLAIR', 'T1ce', 'T1', 'T2']
MASK = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0],
                 [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
N, M, K = 8, 4, 2
ENC_B = np.full(M, 1_466_496 * 4.0)
DEC_B = 2_540_732 * 4.0


def _walk(o, p=''):
    yield p, o
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f'{p}.{k}' if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:400]):
            yield from _walk(v, f'{p}[{i}]')


def load(path):
    with open(path) as f:
        doc = json.load(f)
    node = None
    for p, v in _walk(doc):
        if p.split('.')[-1] == 'rounds' and isinstance(v, (dict, list)):
            node = v
            break
    items = (sorted(node.items(), key=lambda kv: int(kv[0]))
             if isinstance(node, dict) else list(enumerate(node or [])))
    ev_t, ev_a, T = [], [], 0
    for t, r in items:
        if not isinstance(r, dict):
            continue
        T = max(T, int(t) + 1)
        if 'dice_matrix' in r:
            d = np.asarray(r['dice_matrix'], float)
            d = d.mean(1) if d.ndim == 2 else d
            if d.shape[0] == N and np.isfinite(d).all():
                ev_t.append(int(t))
                ev_a.append(d)              # performance a_{i,t}
    if not ev_t:
        return None
    return dict(path=path, T=T, ev_t=np.array(ev_t), ev_a=np.array(ev_a))


class Ctx:
    """Serves loss = -performance. step=True reproduces replay_exp2.py."""

    def __init__(self, run, step=True):
        self.run, self.step = run, step
        self.n_samples = {k: 1 for k in range(N)}
        self.encoder_bytes = {m: float(ENC_B[m]) for m in range(M)}
        self.t = 0

    def perf(self, t):
        et, ea = self.run['ev_t'], self.run['ev_a']
        if self.step:
            i = int(np.searchsorted(et, t, side='right') - 1)
            return ea[max(i, 0)]
        if t <= et[0]:
            return ea[0]
        if t >= et[-1]:
            return ea[-1]
        j = int(np.searchsorted(et, t))
        lo, hi = et[j - 1], et[j]
        f = (t - lo) / (hi - lo)
        return ea[j - 1] * (1 - f) + ea[j] * f

    def set_round(self, t):
        self.t = t

    def local_loss(self, ids):
        v = -self.perf(self.t)
        return {int(k): float(v[int(k)]) for k in ids}

    def shapley(self, k):
        return {m: 1.0 for m in range(M) if MASK[k, m]}


def run(run_, step, tau=1.0):
    sel = make_selector('mmic', MASK, K, seed=42, tau=tau)
    log = CoverageLog(MASK, DEC_B, ENC_B)
    ctx = Ctx(run_, step=step)
    zero_rounds = 0
    for t in range(run_['T']):
        ctx.set_round(t)
        plan = sel.plan_round(t, ctx)
        before = {k: sel.perf.get(k) for k in plan.train}
        losses = ctx.local_loss(plan.train)
        sel.observe(t, plan, losses_after=losses)
        moved = any(before[k] is not None
                    and abs(-losses[k] - before[k]) > 1e-12 for k in plan.train)
        zero_rounds += (not moved)
        log.record(t, plan)
    return sel, log.summary(), zero_rounds / run_['T']


def main(paths):
    runs = [r for r in (load(p) for p in paths) if r]
    if not runs:
        raise SystemExit('no usable metrics.json')
    cnt = MASK.sum(1)

    print('=' * 92)
    print('IS MMiC ≈ UNIFORM GENUINE, OR A REPLAY ARTEFACT?')
    print('=' * 92)
    for r in runs:
        name = r['path'].split('/')[-2][:46]
        print(f'\n{name}')
        gap = int(np.diff(r['ev_t']).min()) if len(r['ev_t']) > 1 else 0
        print(f'  evaluations every {gap} rounds -> between them the replayed'
              f' performance is CONSTANT')

        for step, label in ((True, 'STEP  (as replayed)'),
                            (False, 'INTERPOLATED RATE')):
            sel, s, zfrac = run(r, step)
            phi, T_ = sel.phi, sel.T
            ratio = phi / np.maximum(T_, 1)
            prob = sel._probs()
            print(f'\n  {label}')
            print(f'    rounds with NO performance change: {zfrac:6.1%}'
                  + ('   <- alpha is zero, Eq (7) cannot fire'
                     if zfrac > 0.5 else ''))
            print(f'    phi        ' + ' '.join(f'{v:5.0f}' for v in phi))
            print(f'    times sel. ' + ' '.join(f'{v:5.0f}' for v in T_))
            print(f'    phi/T      ' + ' '.join(f'{v:5.2f}' for v in ratio))
            print(f'    prob(i)    ' + ' '.join(f'{v:5.3f}' for v in prob)
                  + f'   uniform = {1/N:.3f}')
            print(f'    max |prob - uniform| = {np.abs(prob - 1/N).max():.4f}'
                  f'   T1ce coverage = {s["encoder_updates"][1]}'
                  f'   client Jain = {s["client_jain"]:.3f}')
            # the paper's own claim, tested
            a = cnt - cnt.mean()
            b = ratio - ratio.mean()
            d = np.sqrt((a @ a) * (b @ b))
            rr = float(a @ b / d) if d > 0 else 0.0
            print(f'    paper claims BPI favours modality-COMPLETE clients:'
                  f'  corr(modalities, phi/T) = {rr:+.3f}')

    print('\n' + '=' * 92)
    print('HOW TO READ THIS')
    print('=' * 92)
    print('  STEP run shows a high "no performance change" fraction AND a flat')
    print('  prob(i)  ->  ARTEFACT. MMiC cannot be replayed from evaluations')
    print('  spaced 10 rounds apart, because Eq (7) differentiates the series.')
    print('  Report MMiC from a LIVE run, or from interpolated rates with the')
    print('  assumption declared. Do not report the step-run number.')
    print()
    print('  INTERPOLATED run still flat  ->  GENUINE. MMiC really does not')
    print('  concentrate selection in this federation. That is reportable, and')
    print('  it is a finding about the method rather than about your logs.')
    print()
    print('  Either way, pow-d is unaffected: it ranks by the LEVEL of the')
    print('  loss, which a step function preserves exactly. Only MMiC uses a')
    print('  difference, and only MMiC is exposed to this.')


if __name__ == '__main__':
    args = sys.argv[1:]
    paths = sorted(p for a in args for p in glob.glob(a)) or sorted(
        glob.glob('results/*K2_150r_seed*/metrics.json'))
    if not paths:
        raise SystemExit('usage: python diagnose_mmic.py <metrics.json> ...')
    main(paths)
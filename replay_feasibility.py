# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import json
import sys

import numpy as np

MODS = ['FLAIR', 'T1ce', 'T1', 'T2']
MASK_A = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0],
                   [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
N, K = 8, 2


def _walk(o, p=''):
    yield p, o
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f'{p}.{k}' if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:400]):
            yield from _walk(v, f'{p}[{i}]')


def _rounds_container(doc):
    for path, v in _walk(doc):
        if path.split('.')[-1] == 'rounds' and isinstance(v, (dict, list)):
            return v
    return doc.get('rounds')


def dice_series(doc, key='dice_matrix'):
    """-> (rounds, 8) mean-over-region Dice at each evaluated round."""
    node = _rounds_container(doc)
    if node is None:
        return None, None
    items = (sorted(node.items(), key=lambda kv: int(kv[0]))
             if isinstance(node, dict) else list(enumerate(node)))
    ts, rows = [], []
    for t, r in items:
        if isinstance(r, dict) and key in r:
            m = np.asarray(r[key], float)
            if m.shape[0] == N:
                ts.append(int(t))
                rows.append(m.mean(1) if m.ndim == 2 else m)
    if not rows:
        return None, None
    return np.array(ts), np.array(rows)


def spearman(a, b):
    def mid(x):
        x = np.asarray(x, float)
        o = np.argsort(x)
        r = np.empty(len(x))
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    a, b = mid(a) - np.mean(mid(a)), mid(b) - np.mean(mid(b))
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def analyse(path, scarce_name='T1ce', mask=MASK_A):
    with open(path) as f:
        doc = json.load(f)
    ts, dice = dice_series(doc)
    if ts is None:
        print(f'!! {path}: no dice_matrix found in the rounds container.')
        return None
    loss = 1.0 - dice                      # monotone: ranking is what matters
    sc = MODS.index(scarce_name)
    pool = [k for k in range(N) if mask[k, sc]]

    print(f'\n{"=" * 78}\n{path.split("/")[-2][:70]}\n{"=" * 78}')
    print(f'    {len(ts)} evaluation rounds: {ts[0]} .. {ts[-1]}')
    print(f'    scarce = {scarce_name}, held by {[f"C{k+1}" for k in pool]}')

    # ---- T1 rank stability
    print(f'\n    T1  RANK STABILITY (Spearman vs the final evaluation)')
    print(f'        {"round":>7}{"rho":>8}   client order, lowest loss first')
    rhos = []
    for i, t in enumerate(ts):
        r = spearman(loss[i], loss[-1])
        rhos.append(r)
        order = ' '.join(f'C{k+1}' for k in np.argsort(loss[i]))
        print(f'        {t:>7}{r:>8.3f}   {order}')
    print(f'        mean rho over all but the last = {np.mean(rhos[:-1]):+.3f}')

    # ---- T2 the decisive cell
    print(f'\n    T2  ARE THE SCARCE HOLDERS IN THE BOTTOM TWO BY LOSS?')
    print('        (pow-d at d=2K can never select the two lowest-loss sites)')
    hits = 0
    for i, t in enumerate(ts):
        order = list(np.argsort(loss[i]))
        rk = sorted(order.index(k) + 1 for k in pool)
        ok = rk == [1, 2]
        hits += ok
        print(f'        round {t:>3}: scarce holders at loss ranks {rk}'
              f'   {"-> unselectable" if ok else "-> REACHABLE"}')
    frac = hits / len(ts)
    print(f'        unselectable in {hits}/{len(ts)} evaluated rounds '
          f'({frac:.0%})')

    # ---- T3 margin
    print(f'\n    T3  MARGIN TO THE RANK-2 BOUNDARY')
    wob = float(np.mean(np.std(loss, axis=0)))
    gaps = []
    for i in range(len(ts)):
        third = np.sort(loss[i])[2]                # 3rd lowest loss
        g = third - max(loss[i][k] for k in pool)  # >0 => both holders below it
        gaps.append(g)
    gaps = np.array(gaps)
    print(f'        between-round wobble in client loss  = {wob:.4f}')
    print(f'        mean gap to the 3rd-lowest-loss site = {gaps.mean():+.4f}'
          f'  ({gaps.mean()/wob if wob else float("nan"):+.1f} wobbles)')
    print(f'        worst single round                   = {gaps.min():+.4f}')
    return dict(path=path, frac=frac, rho=float(np.mean(rhos[:-1])),
                gap=float(gaps.mean()), worst=float(gaps.min()), wob=wob)


def main(paths):
    res = [r for r in (analyse(p) for p in paths) if r]
    if not res:
        raise SystemExit('nothing analysed')
    print(f'\n{"=" * 78}\nVERDICT — IS A LOGGED-LOSS REPLAY DEFENSIBLE?\n{"=" * 78}')
    frac = float(np.mean([r['frac'] for r in res]))
    rho = float(np.mean([r['rho'] for r in res]))
    worst = float(np.min([r['worst'] for r in res]))
    print(f'    rank stability (mean rho vs final)      {rho:+.3f}')
    print(f'    scarce holders unselectable, all seeds  {frac:.0%} of rounds')
    print(f'    worst-round margin                      {worst:+.4f}')
    print()
    if rho > 0.8 and frac > 0.9 and worst > 0:
        print('    YES. The client loss ranking is a property of the data, not')
        print('    of the trajectory: it holds from the first evaluation to the')
        print('    last, and the scarce holders never leave the unselectable')
        print('    pair. A replay over logged losses measures the same selection')
        print('    process a live run would. Report it as a REPLAY, state the')
        print('    self-correction caveat, and you do not need the GPU for the')
        print('    coverage and staleness claims.')
    elif rho > 0.8 and frac > 0.6:
        print('    QUALIFIED YES. The ranking is broadly stable but the scarce')
        print('    holders are reachable in some rounds. Replay gives the right')
        print('    DIRECTION but the exact coverage count is not trustworthy.')
        print('    Report the mechanism and the direction; do not quote an exact')
        print('    0/150 without a live run.')
    else:
        print('    NO. The ranking moves during training, so a replay would')
        print('    measure a trajectory that never existed. You need the live')
        print('    run. This is itself worth reporting: it means loss-based')
        print('    selection is NOT structurally determined in your setting.')
    print()
    print('    In every case, replay cannot show whether starvation is')
    print('    SELF-CORRECTING — the feedback where a stale encoder degrades')
    print('    its holders, raises their loss, and makes pow-d pick them after')
    print('    all. State that limit explicitly. It is the strongest reason to')
    print('    run the GPU arm later, and it could cut either way.')


if __name__ == '__main__':
    args = sys.argv[1:]
    paths = sorted(p for a in args for p in glob.glob(a)) or sorted(
        glob.glob('results/*K2_150r_seed*/metrics.json'))
    if not paths:
        raise SystemExit('usage: python replay_feasibility.py <metrics.json> ...')
    main(paths)
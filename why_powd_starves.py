# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
from math import comb

import numpy as np

MODS = ['FLAIR', 'T1ce', 'T1', 'T2']

MASK_A = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0],
                   [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
MASK_B = MASK_A[:, [1, 0, 2, 3]]

# measured per-client macro Dice, shared 50-case test, K = 8, seed 42
MACRO = {'A': np.array([71.874, 72.260, 57.334, 54.253,
                        55.382, 48.943, 55.573, 43.583]),
         'B': np.array([75.216, 73.642, 74.877, 63.698,
                        69.401, 50.277, 63.683, 43.864])}

SIGMA = 0.01        # per-round loss noise in DryContext, for the margin scale
N, M, K = 8, 4, 2


def rule(c='='):
    print(c * 84)


# ----------------------------------------------------------------- helpers
def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def ranks(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def exact_perm_p(x, y, stat=pearson):
    """Exact one-sided permutation test. n = 8 so 8! = 40320 — enumerable."""
    obs = stat(x, y)
    y = np.asarray(y, float)
    ge = sum(1 for p in itertools.permutations(range(len(y)))
             if stat(x, y[list(p)]) >= obs)
    return obs, ge / np.math.factorial(len(y)) if hasattr(np, 'math') \
        else (obs, ge / 40320)


def _fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def perm_p(x, y, stat=pearson):
    obs = stat(x, y)
    y = np.asarray(y, float)
    ge = sum(1 for p in itertools.permutations(range(len(y)))
             if stat(x, y[list(p)]) >= obs)
    return obs, ge / _fact(len(y))


def unselectable(loss, d, K=K):
    """Exact: for each client, the best margin over every candidate set of
    size d. Client k enters S iff at most K-1 members of A\\{k} have a
    strictly larger loss, i.e. loss[k] >= (K-th largest loss among the others).
    A non-positive best margin means k can never be selected, for any draw."""
    best = {k: -np.inf for k in range(len(loss))}
    for A in itertools.combinations(range(len(loss)), d):
        for k in A:
            o = sorted(loss[j] for j in A if j != k)
            best[k] = max(best[k], loss[k] - o[-K])
    return best


# ------------------------------------------------------------------- C1
def claim1():
    rule()
    print('C1  POOL SCARCITY AND LOW LOSS SHARE A CAUSE: MODALITY RICHNESS')
    rule()
    cnt = MASK_A.sum(1).astype(float)          # row sums, same in both scenarios
    print(f'    {"client":<14}' + ''.join(f'{"C%d" % (k + 1):>8}' for k in range(N)))
    print(f'    {"modalities":<14}' + ''.join(f'{int(c):8d}' for c in cnt))
    ok = True
    for s in 'AB':
        r, p = perm_p(cnt, MACRO[s])
        rho, pr = perm_p(ranks(cnt), ranks(MACRO[s]))
        print(f'    {"macro Dice " + s:<14}' + ''.join(f'{v:8.1f}' for v in MACRO[s]))
        print(f'{"":<18}pearson r = {r:+.3f}  exact p = {p:.5f}     '
              f'spearman rho = {rho:+.3f}  exact p = {pr:.5f}')
        ok &= (r > 0.8 and p < 0.01)
    print('\n    A site that holds the rare modality holds it because it is')
    print('    well-equipped; being well-equipped is also why it segments best.')
    print('    Scarcity and low loss are therefore confounded BY CONSTRUCTION of')
    print('    any realistic modality manifest, not by an unlucky assignment.')
    assert ok, 'C1 failed'
    return cnt


# ------------------------------------------------------------------- C2
def claim2():
    rule()
    print('C2  AT d = 2K THE TWO LOWEST-LOSS SITES CAN NEVER BE SELECTED')
    rule()
    print('    Exact enumeration of all C(8,d) candidate sets, both scenarios.')
    pred = {}
    for s, mask in (('A', MASK_A), ('B', MASK_B)):
        loss = (100.0 - MACRO[s]) / 100.0        # any monotone map does — see C3
        sc = int(mask.sum(0).argmin())
        pool = [k for k in range(N) if mask[k, sc]]
        order = list(np.argsort(loss))
        print(f'\n    scenario {s}   scarce = {MODS[sc]}   '
              f'pool = {[f"C{k + 1}" for k in pool]}')
        print('      loss order, lowest first: '
              + ' '.join(f'C{k + 1}' for k in order))
        for d in (2 * K, N):
            best = unselectable(loss, d)
            dead = [f'C{k + 1}' for k in range(N) if best[k] <= 0]
            live = [f'C{k + 1}' for k in range(N) if best[k] > 0]
            print(f'      d = {d:<2} unselectable {str(dead):<44} '
                  f'selectable {live}')
            for k in pool:
                print(f'{"":<18}C{k + 1} best-case margin {best[k]:+.4f} '
                      f'= {best[k] / SIGMA:+5.1f} sd of the loss noise')
            if d == 2 * K:
                pred[s] = all(best[k] <= 0 for k in pool)
        # what d = N degenerates to
        best = unselectable(loss, N)
        live = [k for k in range(N) if best[k] > 0]
        cov = sorted({m for k in live for m in range(M) if mask[k, m]})
        print(f'      d = N degenerates to the fixed pair '
              f'{[f"C{k + 1}" for k in live]}, which between them hold '
              f'{[MODS[m] for m in cov]}')
        print(f'{"":<18}=> the other {M - len(cov)} encoder(s) are aggregated in '
              f'0 of T rounds, deterministically.')
    return pred


# ------------------------------------------------------------------- C3
def claim3():
    rule()
    print('C3  THE RESULT USES ONLY THE ORDERING, NOT THE LOSS VALUES')
    rule()
    maps = {'linear   (100-Dice)/100': lambda d: (100.0 - d) / 100.0,
            'log      -log(Dice/100)': lambda d: -np.log(d / 100.0),
            'odds     (100-D)/D':      lambda d: (100.0 - d) / d,
            'squared  ((100-D)/100)^2': lambda d: ((100.0 - d) / 100.0) ** 2}
    for s, mask in (('A', MASK_A), ('B', MASK_B)):
        sc = int(mask.sum(0).argmin())
        pool = [k for k in range(N) if mask[k, sc]]
        outs = set()
        for name, f in maps.items():
            best = unselectable(f(MACRO[s]), 2 * K)
            outs.add(tuple(sorted(k for k in range(N) if best[k] <= 0)))
        print(f'    scenario {s}: {len(maps)} different loss maps -> '
              f'{len(outs)} distinct unselectable set(s)')
        for o in outs:
            print(f'{"":<18}{[f"C{k + 1}" for k in o]}   '
                  f'scarce pool inside: {all(k in o for k in pool)}')
    print('\n    So the dry run\'s stand-in loss model is not doing the work.')
    print('    Only the client RANKING is, and that ranking is measured.')


# ---------------------------------------------------------------- how rare
def rarity():
    rule()
    print('HOW SPECIAL IS THE REAL CONFIGURATION?')
    rule()
    p = 1 / comb(N, K)
    print(f'    At d = 2K the unselectable set is exactly the {K} lowest-loss '
          f'sites.')
    print(f'    Under a RANDOM assignment of losses to clients, the chance that')
    print(f'    both scarce holders land there is 1 / C({N},{K}) = {p:.2%}.')
    for s, mask in (('A', MASK_A), ('B', MASK_B)):
        loss = (100.0 - MACRO[s]) / 100.0
        sc = int(mask.sum(0).argmin())
        pool = [k for k in range(N) if mask[k, sc]]
        order = list(np.argsort(loss))
        rk = sorted(order.index(k) + 1 for k in pool)
        print(f'    measured, scenario {s}: scarce holders sit at loss ranks '
              f'{rk} of {N}  ->  {"HIT" if rk == [1, 2] else "near miss"}')
    print('\n    A 3.6% event, reached in scenario A and missed by one place in')
    print('    scenario B — which is what an r = +0.90 confound predicts. It is')
    print('    not luck, and it will recur in any manifest where the rare')
    print('    modality sits at the better-equipped sites.')


# ------------------------------------------------------------- prediction
def prediction(pred):
    rule()
    print('PRE-REGISTERED PREDICTION FOR THE LIVE RUNS')
    rule()
    print(f'    {"scenario":<10}{"scarce":<8}{"cpow-d d=2K":<16}{"basis":<40}')
    for s, mask in (('A', MASK_A), ('B', MASK_B)):
        sc = int(mask.sum(0).argmin())
        loss = (100.0 - MACRO[s]) / 100.0
        best = unselectable(loss, 2 * K)
        pool = [k for k in range(N) if mask[k, sc]]
        worst = max(best[k] for k in pool)
        if pred[s]:
            claim, basis = 'exactly 0/150', 'both holders unselectable, by enumeration'
        else:
            claim = 'small, non-zero'
            basis = (f'best margin {worst / SIGMA:+.1f} sd — reachable, but only '
                     f'marginally')
        print(f'    {s:<10}{MODS[sc]:<8}{claim:<16}{basis}')
    print('\n    Scenario A is a theorem under the measured ranking. Scenario B')
    print('    is a coin-flip at the margin, so it is the more informative run:')
    print('    it tests whether the live per-round ranking matches the K=8 one.')
    print('\n    Log the per-round loss ranking. If the live cpow-d run gives a')
    print('    non-zero T1ce in scenario A, the ranking moved during training,')
    print('    and THAT is the finding — report it, do not bury it.')


if __name__ == '__main__':
    print('Why Power-of-Choice starves the scarce encoder')
    print('Exact argument — no sampling, no seed, nothing to get unlucky with.\n')
    claim1()
    print()
    pred = claim2()
    print()
    claim3()
    print()
    rarity()
    print()
    prediction(pred)
    print()
    rule()
    print('BOTTOM LINE FOR THE WRITE-UP')
    rule()
    print('    Do not write "in our simulation pow-d starved T1ce".')
    print('    Write: "under the measured client loss ordering, the T1ce pool is')
    print('    provably outside the reachable set of pi_pow-d for any candidate')
    print('    draw at d = 2K; the encoder is never aggregated." Then cite this')
    print('    enumeration, and the live run as confirmation rather than as')
    print('    evidence.')
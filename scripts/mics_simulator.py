# -*- coding: utf-8 -*-
"""Client-selection simulator — does MICS actually equalise encoder coverage?
"""
import itertools
import numpy as np

MODS = ['FLAIR', 'T1ce', 'T1', 'T2']
T_DEFAULT = 150

# Scenario 1 manifest, columns FLAIR / T1ce / T1 / T2
MANIFEST_A = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0],
                       [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
MANIFEST_B = MANIFEST_A[:, [1, 0, 2, 3]]

# measured per-client benefit of each sequence, summed over regions (Dice pts)
IMPACT_SUM = np.array([35.40, 48.83, 13.89, 8.17])


# ------------------------------------------------------------------ helpers
def utility(mask, mode='impact', impact=IMPACT_SUM):
    """Encoder weighting U_m, normalised to sum 1.

    'uniform'  every encoder equal                         -> deficit_only
    'reach'    n_m * I_m, impact times how many hold it     -> mics_reach
    'impact'   I_m alone                                    -> mics_impact

    The 'reach' variant gives the lowest weight to the scarcest pool, which is
    exactly the pool that needs help. 'impact' removes that inversion.
    """
    M = mask.shape[1]
    if mode == 'uniform':
        u = np.ones(M, dtype=float)
    elif mode == 'reach':
        u = mask.sum(0).astype(float) * impact
    elif mode == 'impact':
        u = np.asarray(impact, dtype=float).copy()
    else:
        raise ValueError(mode)
    return u / u.sum()


POLICY_U = {'deficit_only': 'uniform', 'mics_reach': 'reach', 'mics_impact': 'impact'}
GREEDY_POLICIES = tuple(POLICY_U)


def jain(x):
    x = np.asarray(x, dtype=float)
    return x.sum() ** 2 / (len(x) * (x ** 2).sum()) if (x ** 2).sum() > 0 else 1.0


def longest_gap(selected, k, T):
    """Longest run of consecutive rounds in which client k was not selected."""
    best = run = 0
    for t in range(T):
        if k in selected[t]:
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


def g_concave(x):
    """Non-decreasing concave, g(0)=0. Marginal gain shrinks with contributors."""
    return x / (x + 1.0)


# ------------------------------------------------------------------ objective
def f_value(S, mask, U, delta, stale, t, beta, K):
    """The set function MICS maximises. Monotone submodular by construction."""
    cover = 0.0
    for m in range(mask.shape[1]):
        c = sum(1 for k in S if mask[k, m])
        cover += U[m] * delta[m] * g_concave(c)
    fair = sum(stale[k] / max(t, 1) for k in S) / K
    return cover + beta * fair


def greedy_select(mask, U, delta, stale, t, K, beta, s_max=None):
    N = mask.shape[0]
    S = []
    if s_max is not None:
        forced = [k for k in range(N) if stale[k] >= s_max]
        S = forced[:K]
    while len(S) < K:
        best, bestv = None, -1e18
        base = f_value(S, mask, U, delta, stale, t, beta, K)
        for k in range(N):
            if k in S:
                continue
            v = f_value(S + [k], mask, U, delta, stale, t, beta, K) - base
            if v > bestv:
                best, bestv = k, v
        S.append(best)
    return sorted(S)


def optimal_select(mask, U, delta, stale, t, K, beta):
    """Exhaustive best subset — feasible for small N, used for the ratio only."""
    N = mask.shape[0]
    best, bestv = None, -1e18
    for S in itertools.combinations(range(N), K):
        v = f_value(list(S), mask, U, delta, stale, t, beta, K)
        if v > bestv:
            best, bestv = list(S), v
    return best, bestv


# ------------------------------------------------------------------ policies
def run_policy(mask, K, T, policy, seed=0, beta=0.35, s_max=None, track_ratio=False):
    N, M = mask.shape
    rng = np.random.default_rng(seed)
    Uu = utility(mask, POLICY_U[policy]) if policy in POLICY_U else None
    updates = np.zeros(M, dtype=int)
    stale = np.zeros(N, dtype=int)
    enc_stale = np.zeros(M, dtype=int)
    max_enc_stale = np.zeros(M, dtype=int)
    selected, ratios = [], []
    rr_order = rng.permutation(N)
    rr_ptr = 0

    for t in range(T):
        delta = (t - updates) / max(t, 1)
        delta = np.clip(delta, 0.0, 1.0)

        if policy == 'uniform':
            S = sorted(rng.choice(N, size=K, replace=False).tolist())
        elif policy == 'round_robin':
            S = sorted(int(rr_order[(rr_ptr + i) % N]) for i in range(K))
            rr_ptr = (rr_ptr + K) % N
        elif policy in POLICY_U:
            S = greedy_select(mask, Uu, delta, stale, t, K, beta, s_max)
            if track_ratio:
                gv = f_value(S, mask, Uu, delta, stale, t, beta, K)
                _, ov = optimal_select(mask, Uu, delta, stale, t, K, beta)
                ratios.append(gv / ov if ov > 0 else 1.0)
        else:
            raise ValueError(policy)

        selected.append(S)
        for k in range(N):
            stale[k] = 0 if k in S else stale[k] + 1
        for m in range(M):
            if any(mask[k, m] for k in S):
                updates[m] += 1
                enc_stale[m] = 0
            else:
                enc_stale[m] += 1
                max_enc_stale[m] = max(max_enc_stale[m], enc_stale[m])

    part = np.array([sum(1 for t in range(T) if k in selected[t]) for k in range(N)])
    gaps = np.array([longest_gap(selected, k, T) for k in range(N)])
    return {
        'updates': updates, 'idle_fraction': 1 - updates / T,
        'max_enc_stale': max_enc_stale, 'worst_horizon': int(updates.min()),
        'enc_jain': jain(updates), 'participation': part,
        'client_jain': jain(part), 'max_client_gap': int(gaps.max()),
        'greedy_ratio': float(np.mean(ratios)) if ratios else None,
        'selected': selected,
    }


# ------------------------------------------------------------------ checks
def check_closed_form(mask, K, T=T_DEFAULT, seeds=200):
    """Uniform sampling must reproduce p_m = C(N-n_m,K)/C(N,K)."""
    from math import comb
    N, M = mask.shape
    n = mask.sum(0)
    pred = np.array([comb(N - int(nm), K) / comb(N, K) if N - nm >= K else 0.0 for nm in n])
    obs = np.zeros(M)
    for s in range(seeds):
        obs += run_policy(mask, K, T, 'uniform', seed=1000 + s)['idle_fraction']
    obs /= seeds
    print(f'\n  closed-form check, N={N} K={K}, {seeds} seeds')
    ok = True
    for m in range(M):
        d = abs(obs[m] - pred[m])
        good = d < 0.02
        ok &= good
        print(f'    {MODS[m]:6s} n={n[m]}  predicted {pred[m]:.4f}  observed {obs[m]:.4f}'
              f'   {"ok" if good else "MISMATCH"}')
    return ok


# ------------------------------------------------------------------ main
def compare(mask, K, T=T_DEFAULT, seeds=20, beta=0.35, s_max=12, label=''):
    pols = ['uniform', 'round_robin', 'deficit_only', 'mics_reach', 'mics_impact']
    agg = {p: {'worst': [], 'ejain': [], 'cjain': [], 'gap': [],
               'upd': [], 'ratio': []} for p in pols}
    for s in range(seeds):
        for p in pols:
            r = run_policy(mask, K, T, p, seed=s, beta=beta,
                           s_max=s_max if p in GREEDY_POLICIES else None,
                           track_ratio=(p == 'mics_impact' and s == 0))
            agg[p]['worst'].append(r['worst_horizon'])
            agg[p]['ejain'].append(r['enc_jain'])
            agg[p]['cjain'].append(r['client_jain'])
            agg[p]['gap'].append(r['max_client_gap'])
            agg[p]['upd'].append(r['updates'])
            if r['greedy_ratio'] is not None:
                agg[p]['ratio'].append(r['greedy_ratio'])

    n = mask.sum(0)
    print(f'\n{"=" * 74}\n{label}   N={mask.shape[0]}  K={K}  T={T}  '
          f'pools {dict(zip(MODS, n))}\n{"=" * 74}')
    print(f'{"policy":14}{"worst enc":>11}{"enc Jain":>10}{"client Jain":>13}'
          f'{"max client gap":>16}')
    for p in pols:
        w = np.mean(agg[p]['worst']); ws = np.std(agg[p]['worst'])
        print(f'{p:14}{w:8.1f}±{ws:<4.1f}{np.mean(agg[p]["ejain"]):10.4f}'
              f'{np.mean(agg[p]["cjain"]):13.4f}{np.mean(agg[p]["gap"]):16.1f}')
    print(f'\n  updates per encoder (mean over {seeds} seeds)')
    print(f'  {"policy":14}' + ''.join(f'{m:>9}' for m in MODS))
    for p in pols:
        u = np.mean(np.array(agg[p]['upd']), axis=0)
        print(f'  {p:14}' + ''.join(f'{v:9.1f}' for v in u))
    print(f'\n  encoder weights U_m used by each greedy policy')
    print(f'  {"policy":14}' + ''.join(f'{m:>9}' for m in MODS))
    for p in GREEDY_POLICIES:
        print(f'  {p:14}' + ''.join(f'{v:9.3f}' for v in utility(mask, POLICY_U[p])))

    if agg['mics_impact']['ratio']:
        print(f'\n  greedy / exhaustive-optimal per round: '
              f'{np.mean(agg["mics_impact"]["ratio"]):.4f}   (theory guarantees >= 0.632)')

    base = np.mean(agg['uniform']['worst'])
    print()
    for p in ('round_robin',) + GREEDY_POLICIES:
        v = np.mean(agg[p]['worst'])
        print(f'  worst-encoder horizon, {p:13} vs uniform: {v - base:+.1f} rounds '
              f'({(v / base - 1) * 100:+.1f}%)')

    # the question this run exists to answer: does weighting by impact buy
    # anything over treating every encoder as equally valuable?
    d = np.mean(agg['deficit_only']['worst'])
    for p in ('mics_reach', 'mics_impact'):
        v = np.mean(agg[p]['worst'])
        verdict = 'earns its place' if v > d + 0.5 else ('ties' if v > d - 0.5 else 'LOSES')
        print(f'  {p:13} vs deficit_only (worst encoder): {v - d:+.1f} rounds  -> {verdict}')

    # worst encoder is not the only thing a weighted rule could improve; a rule
    # that weights by impact should protect the high-impact encoder specifically.
    print(f'\n  impact-weighted coverage  sum_m I_m * updates_m / (T * sum_m I_m)')
    for p in pols:
        u = np.mean(np.array(agg[p]['upd']), axis=0)
        print(f'  {p:14}{float(u @ IMPACT_SUM) / (T * IMPACT_SUM.sum()):9.4f}')
    return agg


def beta_sweep(mask, K, T=T_DEFAULT, seeds=10):
    print(f'\n{"=" * 74}\nbeta sweep — coverage against client fairness\n{"=" * 74}')
    print(f'{"beta":>6}{"worst enc":>12}{"enc Jain":>11}{"client Jain":>13}{"max gap":>10}')
    for b in (0.0, 0.15, 0.35, 0.6, 1.0, 2.0):
        w, ej, cj, g = [], [], [], []
        for s in range(seeds):
            r = run_policy(mask, K, T, 'mics_impact', seed=s, beta=b, s_max=12)
            w.append(r['worst_horizon']); ej.append(r['enc_jain'])
            cj.append(r['client_jain']); g.append(r['max_client_gap'])
        print(f'{b:6.2f}{np.mean(w):12.1f}{np.mean(ej):11.4f}'
              f'{np.mean(cj):13.4f}{np.mean(g):10.1f}')


def make_manifest(N, pool_sizes, seed=0):
    """Random manifest with the given pool sizes; every client keeps >= 1 sequence."""
    rng = np.random.default_rng(seed)
    M = len(pool_sizes)
    for _ in range(500):
        mask = np.zeros((N, M), dtype=int)
        for m, n in enumerate(pool_sizes):
            mask[rng.choice(N, size=n, replace=False), m] = 1
        if (mask.sum(1) > 0).all():
            return mask
    raise RuntimeError('could not build a manifest with those pool sizes')


if __name__ == '__main__':
    print('MICS simulator — selection only, no training')

    ok = check_closed_form(MANIFEST_A, K=2)
    ok &= check_closed_form(MANIFEST_A, K=4)
    if not ok:
        raise SystemExit('\nclosed-form check failed — the sampler is wrong')

    compare(MANIFEST_A, K=2, label='Scenario 1  (T1ce scarce)')
    compare(MANIFEST_B, K=2, label='Scenario 2  (FLAIR scarce)')
    compare(MANIFEST_A, K=4, label='Scenario 1, K=4')

    beta_sweep(MANIFEST_A, K=2)

    print(f'\n{"=" * 74}\ngeneralisation: random manifests, N=16\n{"=" * 74}')
    for i, pools in enumerate([(14, 4, 10, 8), (12, 6, 10, 8), (9, 8, 9, 8)]):
        mk = make_manifest(16, pools, seed=i)
        compare(mk, K=4, seeds=10, label=f'random manifest {i + 1}')

    # ---- the discriminating test -------------------------------------------
    # In BOTH of his real scenarios the scarce pool is also a high-impact pool
    # (T1ce 48.83, FLAIR 35.40), so weighting by impact and weighting by
    # scarcity point the same way and cannot be told apart. These two manifests
    # break the correlation: the scarce pool is T2, the LOWEST-impact sequence.
    #
    # Prediction if U_m is real and not a proxy for 1/n:
    #   mics_impact should LOSE to deficit_only on worst-encoder coverage
    #   (it is deliberately spending less on the starved-but-cheap encoder)
    #   and WIN on impact-weighted coverage. Both must hold. If it loses both,
    #   U_m is noise. If it wins both, something is wrong with the setup.
    print(f'\n{"=" * 74}\nDISCRIMINATING TEST — scarce pool is the LOW-impact one\n{"=" * 74}')
    print('  scarce = T2 (I=8.17). impact weighting should give up worst-encoder')
    print('  coverage here and buy impact-weighted coverage with it.')
    compare(make_manifest(16, (12, 10, 9, 3), seed=41), K=4, seeds=10,
            label='adversarial 1  (T2 scarce, n=3)')
    compare(make_manifest(16, (13, 11, 10, 2), seed=42), K=4, seeds=10,
            label='adversarial 2  (T2 scarce, n=2)')

    print(f'\n{"=" * 74}\nhow to read this\n{"=" * 74}')
    print('1. worst-encoder column. all three greedy policies must beat uniform')
    print('   and round_robin there, or none of this helps in training either.')
    print('2. mics_impact vs deficit_only. deficit_only is the same rule with')
    print('   U_m flat, so it is the ablation of the impact weights. if impact')
    print('   never wins on either worst-encoder or impact-weighted coverage,')
    print('   drop U_m from the rule — the contribution is cleaner without a')
    print('   term that has to be estimated and does not earn its keep.')
    print('3. mics_reach is the n_m * I_m variant. it is expected to LOSE: on')
    print('   Scenario 2 it hands the scarce pool the smallest weight.')
    print('4. the two adversarial manifests are the only place in this run where')
    print('   scarcity and impact disagree. whatever happens there is what U_m')
    print('   is actually doing; everywhere else it is confounded with 1/n.')

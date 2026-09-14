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
def utility(mask, impact=IMPACT_SUM):
    """U_m proportional to n_m * I_m, normalised to sum 1."""
    n = mask.sum(0).astype(float)
    u = n * impact
    return u / u.sum()
 
 
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
    U = utility(mask)
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
        elif policy in ('mics', 'deficit_only'):
            Uu = np.ones(M) / M if policy == 'deficit_only' else U
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
    pols = ['uniform', 'round_robin', 'deficit_only', 'mics']
    agg = {p: {'worst': [], 'ejain': [], 'cjain': [], 'gap': [],
               'upd': [], 'ratio': []} for p in pols}
    for s in range(seeds):
        for p in pols:
            r = run_policy(mask, K, T, p, seed=s, beta=beta,
                           s_max=s_max if p in ('mics', 'deficit_only') else None,
                           track_ratio=(p == 'mics' and s == 0))
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
    if agg['mics']['ratio']:
        print(f'\n  greedy / exhaustive-optimal per round: '
              f'{np.mean(agg["mics"]["ratio"]):.4f}   (theory guarantees >= 0.632)')
 
    base = np.mean(agg['uniform']['worst'])
    for p in ('round_robin', 'deficit_only', 'mics'):
        v = np.mean(agg[p]['worst'])
        print(f'  worst-encoder horizon, {p:13} vs uniform: {v - base:+.1f} rounds '
              f'({(v / base - 1) * 100:+.1f}%)')
    return agg
 
 
def beta_sweep(mask, K, T=T_DEFAULT, seeds=10):
    print(f'\n{"=" * 74}\nbeta sweep — coverage against client fairness\n{"=" * 74}')
    print(f'{"beta":>6}{"worst enc":>12}{"enc Jain":>11}{"client Jain":>13}{"max gap":>10}')
    for b in (0.0, 0.15, 0.35, 0.6, 1.0, 2.0):
        w, ej, cj, g = [], [], [], []
        for s in range(seeds):
            r = run_policy(mask, K, T, 'mics', seed=s, beta=b, s_max=12)
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
 
    print('\nread the worst-encoder column. if MICS does not beat both uniform')
    print('and round_robin there, it will not help in training either.')
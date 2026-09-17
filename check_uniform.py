# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import itertools
import json
import sys

import numpy as np

N, K, T_DEFAULT = 8, 2, 150
MC = 200_000


# ------------------------------------------------------------------ loading
SEL_KEYS = ('selected_clients', 'selected', 'clients', 'participants',
            'client_ids', 'selection', 'train_clients', 'chosen')
SIZE_KEYS = ('n_samples', 'client_sizes', 'dataset_sizes', 'num_samples',
             'n_cases', 'sizes', 'n_train')


def _walk(obj, path=''):
    """Yield (path, value) for every node, so we can find the field whatever
    it is called and however deeply it is nested."""
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f'{path}.{k}' if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:400]):
            yield from _walk(v, f'{path}[{i}]')


def find_selections(doc):
    """Return (rounds x K) int array of 0-based client ids, and the path used."""
    best = None
    for path, v in _walk(doc):
        if not isinstance(v, list) or len(v) < 10:
            continue
        rows = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == K \
                    and all(isinstance(x, (int, float)) for x in item):
                rows.append([int(x) for x in item])
            elif isinstance(item, dict):
                for key in SEL_KEYS:
                    if key in item and isinstance(item[key], (list, tuple)) \
                            and len(item[key]) == K:
                        rows.append([int(x) for x in item[key]])
                        break
                else:
                    rows = []
                    break
            else:
                rows = []
                break
        if rows and len(rows) >= 10:
            if best is None or len(rows) > len(best[0]):
                best = (rows, path)
    if best is None:
        return None, None
    rows = np.array(best[0], int)
    if rows.min() == 1 and rows.max() == N:          # 1-based -> 0-based
        rows = rows - 1
    return rows, best[1]


def find_sizes(doc):
    for path, v in _walk(doc):
        leaf = path.split('.')[-1].split('[')[0]
        if leaf in SIZE_KEYS:
            if isinstance(v, dict) and len(v) == N:
                try:
                    return np.array([float(v[k]) for k in sorted(v, key=str)]), path
                except Exception:
                    pass
            if isinstance(v, list) and len(v) == N \
                    and all(isinstance(x, (int, float)) for x in v):
                return np.array(v, float), path
    return None, None


# ------------------------------------------------------------------- stats
def dispersion(part, T):
    E = T * K / N
    return float((((part - E) ** 2) / E).sum())


def mc_null(T, reps=MC, seed=0):
    """Null distribution of the dispersion statistic under genuine uniform
    sampling of K distinct clients per round."""
    rng = np.random.default_rng(seed)
    pairs = np.array(list(itertools.combinations(range(N), K)))
    E = T * K / N
    out = np.empty(reps)
    chunk = 5000
    for a in range(0, reps, chunk):
        b = min(a + chunk, reps)
        draw = pairs[rng.integers(len(pairs), size=(b - a, T))].reshape(b - a, T * K)
        cnt = np.stack([(draw == k).sum(1) for k in range(N)], 1)
        out[a:b] = (((cnt - E) ** 2) / E).sum(1)
    return out


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def main(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        sel, where = find_selections(doc)
        if sel is None:
            print(f'!! {p}: could not locate a per-round selection list.')
            print('   Open it and tell me the key name; the parser is at the '
                  'top of this file.')
            continue
        sizes, spath = find_sizes(doc)
        part = np.array([(sel == k).sum() for k in range(N)])
        runs.append(dict(path=p, T=len(sel), part=part, sizes=sizes,
                         where=where, spath=spath))
        print(f'loaded {p}\n       selections from "{where}"  '
              f'{len(sel)} rounds'
              + (f'   sizes from "{spath}"' if sizes is not None else
                 '   (no client sizes found)'))
    if not runs:
        raise SystemExit('nothing to test')

    print('\n' + '=' * 78)
    print('1  PER-SEED DISPERSION')
    print('=' * 78)
    print(f'    {"run":<34}' + ''.join(f'{"C%d" % (k + 1):>6}' for k in range(N))
          + f'{"stat":>8}{"p":>9}')
    null_cache, stats, ps = {}, [], []
    for r in runs:
        T = r['T']
        if T not in null_cache:
            null_cache[T] = mc_null(T)
        s = dispersion(r['part'], T)
        p = float((null_cache[T] >= s).mean())
        stats.append(s)
        ps.append(p)
        name = r['path'].split('/')[-2][:33] if '/' in r['path'] else r['path'][:33]
        print(f'    {name:<34}' + ''.join(f'{v:6d}' for v in r['part'])
              + f'{s:8.2f}{p:9.4f}')
    print(f'\n    expected {runs[0]["T"] * K / N:.1f} selections per client')

    print('\n' + '=' * 78)
    print('2  COMBINED ACROSS SEEDS   <- this is the one that decides it')
    print('=' * 78)
    if len(runs) < 2:
        print('    only one run given. Pass all three seeds; a single seed at')
        print('    p = 0.003 is a 1-in-300 event and cannot be distinguished')
        print('    from chance.')
    else:
        T = runs[0]['T']
        obs = float(np.sum(stats))
        reps = min(MC, 60_000)
        null = np.stack([mc_null(T, reps, seed=100 + i)
                         for i in range(len(runs))]).sum(0)
        pc = float((null >= obs).mean())
        print(f'    sum of dispersion statistics = {obs:.2f}')
        print(f'    Monte-Carlo p over {reps} joint draws = {pc:.4f}')
        print('    VERDICT:', 'consistent with a uniform sampler'
              if pc > 0.05 else 'NOT consistent with a uniform sampler')

        print('\n' + '=' * 78)
        print('3  DO THE SEEDS AGREE ON WHICH CLIENTS ARE FAVOURED?')
        print('=' * 78)
        print('    Chance gives no agreement. A biased sampler favours the same')
        print('    clients every time.')
        rs = []
        for i, j in itertools.combinations(range(len(runs)), 2):
            r = pearson(runs[i]['part'], runs[j]['part'])
            rs.append(r)
            print(f'    seed pair {i}-{j}: participation correlation r = {r:+.3f}')
        print(f'    mean pairwise r = {np.mean(rs):+.3f}   '
              + ('-> they agree, so it is systematic'
                 if np.mean(rs) > 0.5 else '-> no agreement, so it is chance'))

    print('\n' + '=' * 78)
    print('4  IS IT WEIGHTED BY DATASET SIZE?')
    print('=' * 78)
    any_sizes = False
    for r in runs:
        if r['sizes'] is None:
            continue
        any_sizes = True
        rr = pearson(r['sizes'], r['part'])
        print(f'    {r["path"].split("/")[-2][:40]:<42} '
              f'participation vs |D_k|: r = {rr:+.3f}')
    if not any_sizes:
        print('    No client dataset sizes in metrics.json. Get them with')
        print('    len(client.train_set) per client and check the correlation')
        print('    by hand -- if participation tracks size, the baseline is')
        print('    size-weighted sampling, not uniform sampling, and 4.1P')
        print('    should say so.')

    print('\n' + '=' * 78)
    print('VERDICT')
    print('=' * 78)
    if len(runs) < 2:
        print('    Inconclusive by construction — pass all three seeds.')
        return
    wide = pc <= 0.05
    agree = np.mean(rs) > 0.5
    sized = [pearson(r['sizes'], r['part']) for r in runs if r['sizes'] is not None]
    tracks = bool(sized) and float(np.mean(sized)) > 0.6

    if tracks:
        print('    SIZE-WEIGHTED. Participation tracks |D_k| '
              f'(mean r = {np.mean(sized):+.3f}).')
        print('    This is a legitimate baseline — pow-d draws its candidate set')
        print('    the same way. The problem is only the NAME. Call it')
        print('    "size-weighted random selection" in 4.1P and the deck.')
    elif wide and agree:
        print('    SYSTEMATIC BIAS. The spread is too wide AND the seeds favour')
        print('    the same clients. Find the sampler before Experiment 2, because')
        print('    every coverage number is measured against this baseline.')
        print('    You do NOT need to re-run: an accurate description is enough,')
        print('    and a biased baseline does not weaken the pow-d contrast.')
    elif wide and not agree:
        print('    ONE OUTLIER SEED, NOT A BIASED SAMPLER. The combined spread is')
        print('    wide, but the seeds disagree about which clients are favoured,')
        print('    which is what chance looks like — a sampler bias would repeat.')
        worst = int(np.argmax(ps[::-1]) if False else int(np.argmin(ps)))
        print(f'    Driven by {runs[worst]["path"]} (p = {ps[worst]:.4f}).')
        print('    Action: none to the code. But do NOT quote a single seed\'s')
        print('    encoder coverage as "the" uniform baseline — report the')
        print('    across-seed spread, because that spread is real and this')
        print('    result is exactly why.')
    else:
        print('    CONSISTENT WITH UNIFORM. Change nothing, say nothing.')


if __name__ == '__main__':
    args = sys.argv[1:]
    paths = [p for a in args for p in glob.glob(a)] or glob.glob(
        'results/*K2_150r_seed*/metrics.json')
    if not paths:
        raise SystemExit('usage: python check_uniform.py <metrics.json> ...')
    main(sorted(paths))
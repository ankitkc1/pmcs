
"""Finalise the selection rule. U_m is out.

Run in the same directory as mics_simulator.py.

    pip install numpy matplotlib
    python mics_finalise.py            # numbers only
    python mics_finalise.py --fig      # numbers + dose_response.{png,pdf}

"""
import sys
from math import comb

import numpy as np

from mics_simulator import (MANIFEST_A, MANIFEST_B, MODS, T_DEFAULT,
                            make_manifest, run_policy)

RULE = 'deficit_only'      # MICS with U_m dropped == flat encoder weights
BETA = 0.15                # provisional; section 1 confirms or replaces it
S_MAX = 12
SEEDS = 20


# ------------------------------------------------------------------ helpers
def predicted_worst_coverage(mask, K):
    """Closed-form coverage of the worst-served pool under uniform sampling.

    p_m = C(N-n_m, K) / C(N, K) is the probability pool m is missed in a
    round, so 1 - max_m p_m is the expected coverage fraction of the pool
    that is missed most often. No simulation involved.
    """
    N = mask.shape[0]
    n = mask.sum(0)
    p = [comb(N - int(nm), K) / comb(N, K) if N - int(nm) >= K else 0.0
         for nm in n]
    return 1.0 - max(p)


def summarise(runs):
    upd = np.mean([r['updates'] for r in runs], axis=0)
    return {
        'worst': float(np.mean([r['worst_horizon'] for r in runs])),
        'worst_sd': float(np.std([r['worst_horizon'] for r in runs])),
        'updates': upd,
        'total': float(upd.sum()),
        'enc_jain': float(np.mean([r['enc_jain'] for r in runs])),
        'client_jain': float(np.mean([r['client_jain'] for r in runs])),
        'gap': float(np.mean([r['max_client_gap'] for r in runs])),
        'max_enc_stale': float(np.mean([r['max_enc_stale'].max() for r in runs])),
    }


def both_policies(mask, K, T=T_DEFAULT, seeds=SEEDS, beta=BETA, s_max=S_MAX):
    u = summarise([run_policy(mask, K, T, 'uniform', seed=s) for s in range(seeds)])
    r = summarise([run_policy(mask, K, T, RULE, seed=s, beta=beta, s_max=s_max)
                   for s in range(seeds)])
    return u, r


def manifest_set():
    """Nine configurations spanning balanced to severely imbalanced."""
    return [
        (MANIFEST_A, 2, 'Scn 1  K=2'),
        (MANIFEST_B, 2, 'Scn 2  K=2'),
        (MANIFEST_A, 4, 'Scn 1  K=4'),
        (MANIFEST_B, 4, 'Scn 2  K=4'),
        (make_manifest(16, (14, 4, 10, 8), seed=0), 4, 'rand 1  n_min=4'),
        (make_manifest(16, (12, 6, 10, 8), seed=1), 4, 'rand 2  n_min=6'),
        (make_manifest(16, (9, 8, 9, 8), seed=2), 4, 'rand 3  n_min=8'),
        (make_manifest(16, (12, 10, 9, 3), seed=41), 4, 'adv 1   n_min=3'),
        (make_manifest(16, (13, 11, 10, 2), seed=42), 4, 'adv 2   n_min=2'),
    ]


def rule(ch='='):
    print(ch * 76)


# ------------------------------------------------- 1. beta on a finer grid
def beta_refine(mask, K, T=T_DEFAULT, seeds=SEEDS, s_max=S_MAX):
    grid = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00)
    print()
    rule()
    print('1. BETA -- refined grid')
    rule()
    print('   beta trades client fairness against coverage. The coarse sweep')
    print('   suggested it is NOT a pure cost: a little of it improves coverage,')
    print('   because it breaks the greedy rule out of locking onto one subset.')
    print()
    print(f'{"beta":>6}{"worst enc":>12}{"enc Jain":>11}{"client Jain":>13}'
          f'{"max gap":>10}{"max enc stale":>15}')
    rows = []
    for b in grid:
        s = summarise([run_policy(mask, K, T, RULE, seed=i, beta=b, s_max=s_max)
                       for i in range(seeds)])
        rows.append((b, s))
        print(f'{b:6.2f}{s["worst"]:12.1f}{s["enc_jain"]:11.4f}'
              f'{s["client_jain"]:13.4f}{s["gap"]:10.1f}{s["max_enc_stale"]:15.1f}')
    best = max(rows, key=lambda r: r[1]['worst'])
    print(f'\n   best coverage at beta = {best[0]:.2f}  '
          f'(worst encoder {best[1]["worst"]:.1f}/{T}, '
          f'client Jain {best[1]["client_jain"]:.3f})')
    print('   pick the smallest beta within ~1 round of the peak: it buys the')
    print('   most client fairness for the least coverage.')
    near = [r for r in rows if r[1]['worst'] >= best[1]['worst'] - 1.0]
    pick = max(near, key=lambda r: r[1]['client_jain'])
    print(f'   -> recommended beta = {pick[0]:.2f}')
    return pick[0]


# ---------------------------------------------------- 2. s_max sensitivity
def smax_sweep(mask, K, beta, T=T_DEFAULT, seeds=SEEDS):
    print()
    rule()
    print('2. s_max -- forced inclusion cap')
    rule()
    print('   s_max is the safety net, not the engine. It guarantees no client')
    print('   waits longer than s_max rounds, which in turn caps encoder')
    print('   staleness at s_max+1 for every non-empty pool. If coverage barely')
    print('   moves across this sweep, the deficit term is doing the work and')
    print('   s_max can be set on fairness grounds alone.')
    print()
    print(f'{"s_max":>7}{"worst enc":>12}{"client Jain":>13}{"max gap":>10}'
          f'{"max enc stale":>15}')
    for sm in (4, 6, 8, 12, 16, 24, None):
        s = summarise([run_policy(mask, K, T, RULE, seed=i, beta=beta, s_max=sm)
                       for i in range(seeds)])
        lab = 'off' if sm is None else str(sm)
        flag = ''
        if sm is not None:
            flag = '  ok' if s['max_enc_stale'] <= sm else '  BOUND VIOLATED'
        print(f'{lab:>7}{s["worst"]:12.1f}{s["client_jain"]:13.4f}'
              f'{s["gap"]:10.1f}{s["max_enc_stale"]:15.1f}{flag}')
    N, _ = mask.shape
    print(f'\n   the bound "max encoder staleness <= s_max" is only feasible when')
    print(f'   K(s_max+1) >= N, i.e. s_max >= N/K - 1 = {N / K - 1:.1f} here. Below')
    print('   that, more than K clients hit the cap at once and this')
    print('   implementation ages the surplus, so the guarantee lapses.')


def staleness_tail(mask, K, T=T_DEFAULT, seeds=200):
    """Does the closed form predict the TAIL as well as the mean?

    The rule's deterministic guarantee is on max encoder staleness, so the
    thing to beat is uniform's max staleness, not uniform's mean coverage.
    For i.i.d. misses with probability p_m, the expected longest run of
    consecutive misses in T rounds is about ln(T(1-p_m)) / ln(1/p_m).
    """
    print()
    rule()
    print('2b. DOES THE CLOSED FORM PREDICT THE STALL, NOT JUST THE MEAN?')
    rule()
    N, M = mask.shape
    n = mask.sum(0)
    p = np.array([comb(N - int(nm), K) / comb(N, K) if N - int(nm) >= K else 0.0
                  for nm in n])
    obs = np.zeros(M)
    for s in range(seeds):
        obs += run_policy(mask, K, T, 'uniform', seed=3000 + s)['max_enc_stale']
    obs /= seeds
    print(f'{"":4}{"pool":8}{"n_m":>5}{"p_m":>8}{"predicted":>11}{"observed":>10}')
    for m in range(M):
        pred = (np.log(T * (1 - p[m])) / np.log(1 / p[m])) if 0 < p[m] < 1 else 0.0
        print(f'{"":4}{MODS[m]:8}{n[m]:5d}{p[m]:8.4f}{pred:11.2f}{obs[m]:10.2f}')
    print('\n   if these track, the instrument predicts the quantity the rule')
    print('   bounds -- which is a stronger statement than predicting coverage.')


# ------------------------------------------------------- 3. dose-response
def dose_response(beta, T=T_DEFAULT, seeds=SEEDS):
    print()
    rule()
    print('3. DOSE-RESPONSE -- the mechanism claim')
    rule()
    print('   Claim: the rule recovers coverage in proportion to how much')
    print('   uniform loses, and recovers nothing when uniform loses nothing.')
    print('   If a manifest ever showed a large gain at high predicted')
    print('   coverage, the mechanism would not be what is claimed.')
    print()
    print(f'{"manifest":16}{"pools":22}{"pred cov":>10}{"uniform":>9}'
          f'{"rule":>8}{"gain":>8}{"gain %":>9}')
    pts = []
    for mask, K, label in manifest_set():
        u, r = both_policies(mask, K, T=T, seeds=seeds, beta=beta)
        x = predicted_worst_coverage(mask, K)
        gain = r['worst'] - u['worst']
        pts.append({
            'label': label, 'K': K, 'pred': x,
            'u': u['worst'] / T, 'r': r['worst'] / T,
            'gain': gain, 'gain_pct': 100 * gain / u['worst'],
            'u_total': u['total'], 'r_total': r['total'],
            'cap': T * mask.shape[1],
            'u_cj': u['client_jain'], 'r_cj': r['client_jain'],
            'u_gap': u['gap'], 'r_gap': r['gap'],
        })
        print(f'{label:16}{str(list(mask.sum(0))):22}{x:10.3f}'
              f'{u["worst"] / T:9.3f}{r["worst"] / T:8.3f}'
              f'{gain:8.1f}{100 * gain / u["worst"]:9.1f}')

    pts.sort(key=lambda d: d['pred'])
    xs = np.array([d['pred'] for d in pts])
    ys = np.array([d['gain_pct'] for d in pts])
    order_ok = bool(np.all(np.diff(ys) <= 1e-9))
    rho = float(np.corrcoef(xs, ys)[0, 1])
    print(f'\n   Spearman-style check: gain is monotone non-increasing in')
    print(f'   predicted coverage?  {order_ok}    Pearson r = {rho:+.3f}')
    if not order_ok:
        bad = [(pts[i]['label'], pts[i + 1]['label'])
               for i in range(len(ys) - 1) if ys[i + 1] > ys[i] + 1e-9]
        print(f'   inversions: {bad}')
        print('   an inversion is not fatal -- K differs across rows -- but it')
        print('   means predicted coverage alone does not order the gain.')
    return pts


# --------------------------------------------------- 4. numbers for the deck
def deck_numbers(pts, T=T_DEFAULT):
    print()
    rule()
    print('4. NUMBERS FOR THE DECK')
    rule()
    for d in pts:
        if d['label'].startswith('Scn 1  K=2'):
            s1 = d
            break
    else:
        s1 = pts[0]
    print(f'   {s1["label"]}, identical communication budget '
          f'(same K, same rounds, same bytes):')
    print(f'     aggregate encoder coverage   '
          f'{100 * s1["u_total"] / s1["cap"]:5.1f}%  ->  '
          f'{100 * s1["r_total"] / s1["cap"]:5.1f}%  of the achievable maximum')
    print(f'     worst-served encoder         '
          f'{100 * s1["u"]:5.1f}%  ->  {100 * s1["r"]:5.1f}%  of rounds')
    print(f'     longest a client waits       '
          f'{s1["u_gap"]:5.1f}   ->  {s1["r_gap"]:5.1f}   rounds')
    print(f'     client participation Jain    '
          f'{s1["u_cj"]:5.3f}   ->  {s1["r_cj"]:5.3f}')
    print()
    print('   Say the fourth line out loud before anyone asks. The rule buys')
    print('   encoder coverage with client equality -- but note line three:')
    print('   participation becomes unequal and MORE regular at the same time.')
    print('   Nobody is abandoned; the waiting is bounded where uniform\'s is not.')


# ------------------------------------------------------------------ figure
def make_figure(pts, beta, outdir='.'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_U, C_R, C_G = '#B23A48', '#1F6F8B', '#8A8F98'
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

    xs = np.array([d['pred'] for d in pts])
    u = np.array([d['u'] for d in pts])
    r = np.array([d['r'] for d in pts])
    gp = np.array([d['gain_pct'] for d in pts])

    a = ax[0]
    a.plot(xs, 100 * u, 'o-', color=C_U, lw=1.8, ms=6, label='uniform (random)')
    a.plot(xs, 100 * r, 's-', color=C_R, lw=1.8, ms=6, label='deficit-aware rule')
    for x0, u0, r0 in zip(xs, u, r):
        a.vlines(x0, 100 * u0, 100 * r0, color=C_G, lw=0.9, zorder=0)
    a.set_xlabel('predicted coverage of the scarcest pool\n'
                 r'$1-\max_m\,\binom{N-n_m}{K}/\binom{N}{K}$   (closed form)')
    a.set_ylabel('worst-encoder coverage (% of rounds)')
    a.set_title('a.  the rule is flat where random collapses', loc='left',
                fontsize=10.5, weight='bold')
    a.set_ylim(35, 104)
    a.legend(frameon=False, fontsize=9, loc='lower right')

    b = ax[1]
    b.plot(xs, gp, 'D-', color=C_R, lw=1.8, ms=6)
    b.axhline(0, color=C_G, lw=0.9, ls='--')
    for d in pts:
        b.annotate(d['label'].split()[0] + ' ' + d['label'].split()[1],
                   (d['pred'], d['gain_pct']), fontsize=7.2,
                   xytext=(4, 5), textcoords='offset points', color='#444')
    b.set_xlabel('predicted coverage of the scarcest pool')
    b.set_ylabel('gain over uniform (%)')
    b.set_title('b.  the gain vanishes when there is nothing to fix', loc='left',
                fontsize=10.5, weight='bold')

    for a_ in ax:
        a_.spines[['top', 'right']].set_visible(False)
        a_.grid(axis='y', color='#E6E6E6', lw=0.8)
        a_.set_axisbelow(True)

    fig.suptitle(f'Deficit-aware selection: effect scales with manifest imbalance '
                 f'  (T={T_DEFAULT}, '
                 rf'$\beta$={beta:.2f}, $s_{{max}}$={S_MAX}, {SEEDS} seeds)',
                 fontsize=10, y=1.005)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{outdir}/dose_response.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'\n   wrote {outdir}/dose_response.png and .pdf')


# -------------------------------------------------------------------- main
if __name__ == '__main__':
    print('finalising the selection rule -- U_m dropped, flat encoder weights')

    beta = beta_refine(MANIFEST_A, K=2)
    smax_sweep(MANIFEST_A, K=2, beta=beta)
    staleness_tail(MANIFEST_A, K=2)
    pts = dose_response(beta)
    deck_numbers(pts)

    if '--fig' in sys.argv:
        make_figure(pts, beta)

    print()
    rule()
    print('WHAT THIS DOES NOT SHOW')
    rule()
    print('   Coverage is a proxy. The mapping from coverage to Dice is not')
    print('   stable in the measured data -- fitting it to the four degradation')
    print('   points gives a constant spanning 0.15 to 2.38. So this settles')
    print('   that the mechanism works, not that Dice improves. That is the')
    print('   training run, and it can fail. Go in knowing that.')

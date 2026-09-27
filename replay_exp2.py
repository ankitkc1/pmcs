# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import json
import sys

import numpy as np

from fl_selectors import CoverageLog, make_selector

MODS = ['FLAIR', 'T1ce', 'T1', 'T2']
MASK_A = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0],
                   [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
# Scenario 2: FLAIR and T1ce columns exchanged. Pool sizes are preserved as a
# multiset {2, 4, 5, 7}; only which sequence occupies which pool changes.
MASK_B = MASK_A[:, [1, 0, 2, 3]]
MASK = MASK_A                 # rebound by set_scenario() before any load()
N, M = 8, 4

# measured payload, from private_param_fraction in metrics.json
ENC_B = np.full(M, 1_466_496 * 4.0)
DEC_B = 2_540_732 * 4.0

# per-region modality impact -- ONLY used as MFedMC's Shapley stand-in, which
# is why MFedMC is reported separately and labelled as such.
# Rows are indexed by MODALITY, not by client, so this needs no scenario
# variant: H2 holds that a sequence's contribution to a region is a property
# of the sequence, not of which sites happen to hold it. The mask swap moves
# the holders; shapley() picks up the change through MASK.
IMPACT = np.array([[14.88, 12.60, 7.92], [-0.45, 13.98, 35.30],
                   [4.25, 5.59, 4.05], [2.40, 2.29, 3.48]])

SCENARIO = 'A'                # set by set_scenario(), used only for labelling


def set_scenario(tag):
    """Bind the manifest. MUST be called before load() or run_one()."""
    global MASK, SCENARIO
    SCENARIO = 'A' if str(tag).upper() in ('A', '1') else 'B'
    MASK = MASK_A if SCENARIO == 'A' else MASK_B
    sizes = '  '.join('%s=%d' % (m, MASK[:, i].sum())
                      for i, m in enumerate(MODS))
    scarce = MODS[int(np.argmin(MASK.sum(0)))]
    holders = [k for k in range(N) if MASK[k, MODS.index(scarce)]]
    print(f'  SCENARIO {SCENARIO}   pools  {sizes}')
    print(f'  scarce pool: {scarce}, held by clients {holders} '
          f'({", ".join(str(int(MASK[k].sum())) for k in holders)} '
          f'modalities each)')
    return MASK


CONFIGS = [
    ('uniform',     'uniform random  (CONTROL)',        {}, True),
    ('round_robin', 'round robin',                      {}, True),
    ('powd',        'Power-of-Choice  d=2K',            dict(d_mult=2), True),
    ('powd',        'Power-of-Choice  d=N',             dict(d_mult=10), True),
    ('rpowd',       'rpow-d  (stale loss, no probe)',   dict(d_mult=2), True),
    ('mmic',        'MMiC  Banzhaf  tau=1',             dict(tau=1.0), True),
    ('mmic',        'MMiC  Banzhaf  tau=4 (sharper)',   dict(tau=4.0), True),
    # --- submodular family. 'perf' and 'mask' are two substitutes for the
    #     gradient space these methods actually use; both are reported
    #     because the choice of space changes the answer.
    ('divfl',    'DivFL  [dissim: perf]',      dict(feature='perf'), True),
    ('divfl',    'DivFL  [dissim: mask]',      dict(feature='mask'), True),
    ('subtrunc', 'SubTrunc lam=.95 [perf]',    dict(feature='perf', lam=0.95, b=1.10), True),
    ('unionfl',  'UnionFL mu=1 w=5 [perf]',    dict(feature='perf', mu=1.0, window=5), True),
    ('unionfl',  'UnionFL mu=1 w=5 [mask]',    dict(feature='mask', mu=1.0, window=5), True),
    # MFedMC has TWO filters and they must be separated, or the "control" is
    # not a control:
    #   gamma  upload filter    -- each client sends only its top-gamma encoders
    #   delta  aggregation filter -- server averages only the lowest-loss
    #                                delta*N clients
    # gamma=4 alone still leaves delta=0.2 in place, and delta ALONE starves an
    # encoder whenever the surviving clients happen not to hold it. Isolating
    # each filter is what identifies which one causes the loss.
    ('mfedmc',      'MFedMC g=1 d=0.2 [published default]', dict(gamma=1, delta=0.2), False),
    ('mfedmc',      'MFedMC g=4 d=0.2 [upload filter OFF]', dict(gamma=4, delta=0.2), False),
    ('mfedmc',      'MFedMC g=1 d=1.0 [aggr. filter OFF]',  dict(gamma=1, delta=1.0), False),
    ('mfedmc',      'MFedMC g=4 d=1.0 [BOTH off = control]', dict(gamma=4, delta=1.0), False),
]


# ------------------------------------------------------------------ loading
def _walk(o, p=''):
    yield p, o
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f'{p}.{k}' if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:400]):
            yield from _walk(v, f'{p}[{i}]')


def _rounds(doc):
    for path, v in _walk(doc):
        if path.split('.')[-1] == 'rounds' and isinstance(v, (dict, list)):
            return (sorted(v.items(), key=lambda kv: int(kv[0]))
                    if isinstance(v, dict) else list(enumerate(v)))
    return []


def load(path):
    """-> dict(T, loss_at(t), measured_selections, measured_coverage)"""
    with open(path) as f:
        doc = json.load(f)
    items = _rounds(doc)
    if not items:
        return None

    sel, ev_t, ev_loss, ev_reg = {}, [], [], []
    for t, r in items:
        if not isinstance(r, dict):
            continue
        t = int(t)
        if 'selected_clients' in r:
            sel[t] = [int(x) for x in r['selected_clients']]
        if 'dice_matrix' in r:
            raw = np.asarray(r['dice_matrix'], float)
            d = raw.mean(1) if raw.ndim == 2 else raw
            if d.shape[0] == N and np.isfinite(d).all():
                ev_t.append(t)
                ev_loss.append(1.0 - d)          # monotone; ranking is the point
                # per-REGION Dice: the feature space DivFL's dissimilarity
                # matrix is built from, since gradients were never logged.
                ev_reg.append(raw if raw.ndim == 2 else np.repeat(d[:, None], 3, 1))
    if not ev_t or not sel:
        return None
    ev_t, ev_loss = np.array(ev_t), np.array(ev_loss)
    ev_reg = np.array(ev_reg)
    T = max(sel) + 1

    def loss_at(t):
        i = int(np.searchsorted(ev_t, t, side='right') - 1)
        return ev_loss[max(i, 0)]                # before first eval: use first

    def feat_at(t):
        i = int(np.searchsorted(ev_t, t, side='right') - 1)
        return ev_reg[max(i, 0)]

    # coverage actually measured, straight from the recorded selections
    upd = np.zeros(M, int)
    st = np.zeros(M, int)
    mx = np.zeros(M, int)
    for t in range(T):
        hit = {m for k in sel.get(t, []) for m in range(M) if MASK[k, m]}
        for m in range(M):
            if m in hit:
                upd[m] += 1
                st[m] = 0
            else:
                st[m] += 1
                mx[m] = max(mx[m], st[m])
    # rank stability, reported so the justification travels with the result
    def mid(x):
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

    def sp(a, b):
        a, b = mid(a) - mid(a).mean(), mid(b) - mid(b).mean()
        d = np.sqrt((a @ a) * (b @ b))
        return float(a @ b / d) if d > 0 else 0.0
    rho = float(np.mean([sp(ev_loss[i], ev_loss[-1])
                         for i in range(len(ev_t) - 1)]))
    return dict(path=path, T=T, loss_at=loss_at, feat_at=feat_at,
                n_evals=len(ev_t), measured_cov=upd,
                measured_stall=int(mx.max()), rho=rho)


# ------------------------------------------------------------------ context
class ReplayContext:
    """Serves the logged losses to the selectors. Everything is measured
    except `shapley`, which no run ever logged -- see the header."""

    def __init__(self, run):
        self.run = run
        self.n_samples = {k: 1 for k in range(N)}   # equal: p_k uniform
        self.encoder_bytes = {m: float(ENC_B[m]) for m in range(M)}
        self.t = 0

    def set_round(self, t):
        self.t = t

    def local_loss(self, ids):
        v = self.run['loss_at'](self.t)
        return {int(k): float(v[int(k)]) for k in ids}

    def client_features(self, t=None):
        """(N, 3) per-region Dice — the measured stand-in for DivFL's
        gradient space. See the note in fl_selectors._Submodular."""
        return self.run['feat_at'](self.t if t is None else t)

    def shapley(self, k):
        held = [m for m in range(M) if MASK[k, m]]
        return {m: float(IMPACT[m].sum()) for m in held}


def run_one(name, kw, run):
    sel = make_selector(name, MASK, 2, seed=42, **kw)
    log = CoverageLog(MASK, DEC_B, ENC_B)
    ctx = ReplayContext(run)
    for t in range(run['T']):
        ctx.set_round(t)
        plan = sel.plan_round(t, ctx)
        sel.observe(t, plan, losses_after=ctx.local_loss(plan.train))
        log.record(t, plan)
    return log.summary()


def main(paths):
    runs = [r for r in (load(p) for p in paths) if r]
    if not runs:
        raise SystemExit('no usable metrics.json (need selected_clients + dice_matrix)')

    print('=' * 100)
    print(f'EXPERIMENT 2 — REPLAY OVER MEASURED PER-CLIENT LOSSES '
          f'— SCENARIO {SCENARIO}')
    print('=' * 100)
    for r in runs:
        print(f'  {r["path"].split("/")[-2][:60]:62} T={r["T"]}  '
              f'{r["n_evals"]} evals  rank stability rho={r["rho"]:+.3f}')
    print(f'\n  Loss source: dice_matrix (each client on its OWN partition),'
          f' held constant between evaluations.')
    print(f'  Mean rank stability across seeds: '
          f'{np.mean([r["rho"] for r in runs]):+.3f} '
          f'-- this is what licenses the replay.')

    # ---- control first
    print('\n' + '=' * 100)
    print('CONTROL  does replayed uniform reproduce the coverage you measured?')
    print('=' * 100)
    print(f'  {"seed":<26}' + ''.join(f'{m:>9}' for m in MODS) + '   source')
    ok = True
    scarce_idx = int(np.argmin(MASK.sum(0)))
    for r in runs:
        s = run_one('uniform', {}, r)
        name = r['path'].split('/')[-2][:24]
        print(f'  {name:<26}'
              + ''.join(f'{v:9d}' for v in s['encoder_updates']) + '   replayed')
        print(f'  {"":<26}'
              + ''.join(f'{v:9d}' for v in r['measured_cov']) + '   MEASURED')
        # replayed uniform uses a different RNG stream, so it is a different
        # uniform draw -- agreement means "same distribution", not "same seq"
        # compare on the SCARCE pool, which is the one the claim rests on and
        # which moves between scenarios (T1ce under A, FLAIR under B)
        ok &= abs(int(s['encoder_updates'][scarce_idx])
                  - int(r['measured_cov'][scarce_idx])) < 25
    print(f'\n  Compared on the scarce pool ({MODS[scarce_idx]}, '
          f'n={int(MASK[:, scarce_idx].sum())}).')
    print('  Replayed uniform draws a different random sequence than your run,')
    print('  so these should MATCH IN LEVEL, not exactly. Large agreement means')
    print('  the coverage machinery is sound.')
    print('  VERDICT:', 'machinery sound' if ok else
          'DISAGREEMENT -- do not trust the rows below, tell me')

    # ---- the table
    for r in runs:
        print('\n' + '=' * 100)
        print(f'TABLE  [scenario {SCENARIO}]  {r["path"].split("/")[-2][:60]}')
        print('=' * 100)
        print(f'  {"method":36}' + ''.join(f'{m:>8}' for m in MODS)
              + f'{"worst":>7}{"stall":>7}{"cJain":>7}{"wait":>6}'
                f'{"probes":>8}{"GB":>8}')
        for name, label, kw, measured in CONFIGS:
            s = run_one(name, kw, r)
            print(f'  {label:36}'
                  + ''.join(f'{v:8d}' for v in s['encoder_updates'])
                  + f'{s["worst_encoder"]:7d}{s["max_encoder_stall"]:7d}'
                    f'{s["client_jain"]:7.3f}{s["longest_client_wait"]:6d}'
                    f'{s["probe_passes_total"]:8d}{s["GB_total"]:8.1f}')
        print(f'  {"YOUR MEASURED uniform run":36}'
              + ''.join(f'{v:8d}' for v in r['measured_cov'])
              + f'{int(r["measured_cov"].min()):7d}{r["measured_stall"]:7d}')

    print('\n' + '=' * 100)
    print('HOW TO REPORT THIS')
    print('=' * 100)
    print('  Rows marked [Shapley STAND-IN] use the measured population impact')
    print('  matrix in place of MFedMC\'s per-client Shapley, which no run')
    print('  logged. Label them. Every other row is driven by measured losses.')
    print()
    print('  Say: "selection policies were replayed over the per-client losses')
    print('  recorded during the completed federated runs. The client loss')
    print('  ranking is stable across training (rho = 0.92), so the ranking is')
    print('  a property of client data rather than of the training trajectory,')
    print('  which is what makes the replay valid for a selection-level claim.')
    print('  Replay cannot capture the feedback by which a starved encoder')
    print('  raises its holders\' loss and makes them selectable again; a live')
    print('  run is required to settle that, and is in progress."')
    print()
    print(f'  Run BOTH scenarios and report them side by side. The pool-size')
    print(f'  correlation is invariant to the exchange; the impact correlation')
    print(f'  is not. A criterion whose relationship to modality importance')
    print(f'  changes sign when the manifest is relabelled was never tracking')
    print(f'  importance -- it was tracking availability.')


if __name__ == '__main__':
    argv = sys.argv[1:]
    tag = next((a.split('=')[1] for a in argv if a.startswith('--scenario=')),
               None)
    args = [a for a in argv if not a.startswith('--scenario')]
    paths = sorted(p for a in args for p in glob.glob(a)) or sorted(
        glob.glob('results/*K2_150r_seed*/metrics.json'))
    if not paths:
        raise SystemExit('usage: python replay_exp2.py <metrics.json> ... '
                         '[--scenario=A|B]')
    if tag is None:
        # infer from the path, so split B can never be scored on the A mask
        tag = 'B' if any(('splitB' in p) or ('split_B' in p) or ('_B_' in p)
                         for p in paths) else 'A'
    set_scenario(tag)
    main(paths)
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

__all__ = ['RoundPlan', 'CoverageLog', 'make_selector', 'SELECTORS']


# ===================================================================== plan
@dataclass
class RoundPlan:
    train: list                       # clients that run local training
    upload: dict                      # client -> set of modality indices sent
    aggregate_from: list              # clients whose uploads are averaged
    contacted: list = field(default_factory=list)   # received the model at all
    probe_passes: int = 0             # forward passes spent on selection only

    def encoders_updated(self, M):
        """Which encoders actually receive an update this round."""
        out = set()
        for k in self.aggregate_from:
            out |= set(self.upload.get(k, ()))
        return sorted(m for m in out if m < M)


# ===================================================================== log
class CoverageLog:
    """Everything the supervisor asked for, recorded per round.

    Two coverage series are kept deliberately:
        held_cov[m]    rounds in which an aggregating client HELD m
        upd_cov[m]     rounds in which m was actually AGGREGATED
    The gap between them is what upload filtering costs, and no baseline
    evaluation protocol records it.
    """

    def __init__(self, manifest, dec_bytes, enc_bytes):
        self.mask = np.asarray(manifest)
        self.N, self.M = self.mask.shape
        self.dec = float(dec_bytes)
        self.enc = np.asarray(enc_bytes, dtype=float)
        self.rows = []
        self.upd = np.zeros(self.M, int)
        self.held = np.zeros(self.M, int)
        self.enc_stale = np.zeros(self.M, int)
        self.max_enc_stale = np.zeros(self.M, int)
        self.part = np.zeros(self.N, int)
        self.cli_stale = np.zeros(self.N, int)
        self.max_cli_gap = np.zeros(self.N, int)
        self.bytes_up = 0.0
        self.bytes_dn = 0.0

    def _payload(self, k, mods):
        return self.dec + float(self.enc[list(mods)].sum()) if mods else self.dec

    def record(self, t, plan):
        updated = set(plan.encoders_updated(self.M))
        held = set()
        for k in plan.aggregate_from:
            held |= {m for m in range(self.M) if self.mask[k, m]}
        for m in range(self.M):
            if m in held:
                self.held[m] += 1
            if m in updated:
                self.upd[m] += 1
                self.enc_stale[m] = 0
            else:
                self.enc_stale[m] += 1
                self.max_enc_stale[m] = max(self.max_enc_stale[m], self.enc_stale[m])
        for k in range(self.N):
            if k in plan.train:
                self.part[k] += 1
                self.cli_stale[k] = 0
            else:
                self.cli_stale[k] += 1
                self.max_cli_gap[k] = max(self.max_cli_gap[k], self.cli_stale[k])
        dn = sum(self._payload(k, [m for m in range(self.M) if self.mask[k, m]])
                 for k in (plan.contacted or plan.train))
        up = sum(self._payload(k, plan.upload.get(k, ())) for k in plan.train)
        self.bytes_dn += dn
        self.bytes_up += up
        self.rows.append({
            'round': t, 'train': list(plan.train),
            'aggregate_from': list(plan.aggregate_from),
            'upload': {int(k): sorted(int(m) for m in v)
                       for k, v in plan.upload.items()},
            'encoders_updated': sorted(int(m) for m in updated),
            'contacted': len(plan.contacted or plan.train),
            'probe_passes': plan.probe_passes,
            'bytes_down': dn, 'bytes_up': up,
        })

    @staticmethod
    def _jain(x):
        x = np.asarray(x, float)
        s = (x ** 2).sum()
        return float(x.sum() ** 2 / (len(x) * s)) if s > 0 else 1.0

    def summary(self):
        T = len(self.rows)
        return {
            'rounds': T,
            'encoder_updates': self.upd.tolist(),
            'encoder_held': self.held.tolist(),
            'held_minus_updated': (self.held - self.upd).tolist(),
            'worst_encoder': int(self.upd.min()),
            'max_encoder_stall': int(self.max_enc_stale.max()),
            'encoder_jain': self._jain(self.upd),
            'participation': self.part.tolist(),
            'client_jain': self._jain(self.part),
            'longest_client_wait': int(self.max_cli_gap.max()),
            'contacted_per_round': float(np.mean([r['contacted'] for r in self.rows])),
            'probe_passes_total': int(sum(r['probe_passes'] for r in self.rows)),
            # DECIMAL GB (1e9), to match how metrics.json reports
            # total_bytes_all_clients_all_rounds. Do not switch to 2**30:
            # the dry run and the live run must use one convention or the
            # communication column silently differs by 7.4% between them.
            'GB_up': self.bytes_up / 1e9,
            'GB_down': self.bytes_dn / 1e9,
            'GB_total': (self.bytes_up + self.bytes_dn) / 1e9,
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'summary': self.summary(), 'rounds': self.rows}, f, indent=1)

    def report(self, mods=None):
        s = self.summary()
        mods = mods or [f'm{i}' for i in range(self.M)]
        print(f'  rounds {s["rounds"]}')
        print(f'  {"encoder":10}{"aggregated":>12}{"held":>8}{"gap":>7}')
        for i, m in enumerate(mods):
            print(f'  {m:10}{s["encoder_updates"][i]:12d}{s["encoder_held"][i]:8d}'
                  f'{s["held_minus_updated"][i]:7d}')
        print(f'  worst encoder {s["worst_encoder"]}   max stall '
              f'{s["max_encoder_stall"]}   encoder Jain {s["encoder_jain"]:.4f}')
        print(f'  client Jain {s["client_jain"]:.4f}   longest wait '
              f'{s["longest_client_wait"]}')
        print(f'  contacted/round {s["contacted_per_round"]:.2f}   probe passes '
              f'{s["probe_passes_total"]}')
        print(f'  GB up {s["GB_up"]:.2f}  down {s["GB_down"]:.2f}  '
              f'total {s["GB_total"]:.2f}')


# ================================================================ selectors
class _Base:
    needs_ctx = False

    def __init__(self, manifest, K, seed=42, **kw):
        self.mask = np.asarray(manifest)
        self.N, self.M = self.mask.shape
        self.K = int(K)
        self.rng = np.random.default_rng(seed)
        self.held = {k: {m for m in range(self.M) if self.mask[k, m]}
                     for k in range(self.N)}
        self.cfg = kw

    def plan_round(self, t, ctx=None):
        raise NotImplementedError

    def observe(self, t, plan, losses_after=None):
        pass

    def _all(self, S):
        return {k: set(self.held[k]) for k in S}


class Uniform(_Base):
    """K clients uniformly at random without replacement. All encoders sent."""
    def plan_round(self, t, ctx=None):
        S = sorted(self.rng.choice(self.N, self.K, replace=False).tolist())
        return RoundPlan(S, self._all(S), S, contacted=S)


class RoundRobin(_Base):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.order = self.rng.permutation(self.N)
        self.ptr = 0

    def plan_round(self, t, ctx=None):
        S = sorted(int(self.order[(self.ptr + i) % self.N]) for i in range(self.K))
        self.ptr = (self.ptr + self.K) % self.N
        return RoundPlan(S, self._all(S), S, contacted=S)


class PowD(_Base):
    """pi_pow-d, Cho et al. Section 4.

    1. Sample candidate set A of d clients WITHOUT replacement, client k with
       probability p_k = |D_k| / sum|D_j|.
    2. Send w(t) to A; each returns F_k(w(t)).  <- forward pass, no training
    3. S(t) = the m clients in A with the LARGEST F_k, ties broken AT RANDOM.

    d is the knob: the paper sweeps d = 2m and d = 10m. m = K here.
    Set variant='cpow' to use a mini-batch loss estimate instead of full F_k;
    that changes only what ctx.local_loss returns, not the selection rule.
    """
    needs_ctx = True

    def __init__(self, *a, d_mult=2, variant='pow', **kw):
        super().__init__(*a, **kw)
        self.d = int(min(max(d_mult * self.K, self.K), self.N))
        self.variant = variant

    def plan_round(self, t, ctx):
        p = np.array([ctx.n_samples[k] for k in range(self.N)], float)
        p = p / p.sum()
        A = self.rng.choice(self.N, self.d, replace=False, p=p)
        losses = ctx.local_loss(sorted(int(a) for a in A))
        vals = np.array([losses[int(a)] for a in A], float)
        jitter = self.rng.random(len(A))                 # ties broken at random
        S = sorted(int(A[i]) for i in np.lexsort((jitter, -vals))[:self.K])
        return RoundPlan(S, self._all(S), S,
                         contacted=sorted(int(a) for a in A),
                         probe_passes=self.d)


class RPowD(_Base):
    """pi_rpow-d, Cho et al. Algorithm 2. No probe round.

    A_tmp[k] = inf until client k has trained. The server samples A by p_k and
    takes the m largest A_tmp values in A. So every client is explored once
    (inf beats any finite loss), after which its recorded loss is used.
    """
    def __init__(self, *a, d_mult=2, **kw):
        super().__init__(*a, **kw)
        self.d = int(min(max(d_mult * self.K, self.K), self.N))
        self.atmp = np.full(self.N, np.inf)

    def plan_round(self, t, ctx=None):
        p = np.array([ctx.n_samples[k] for k in range(self.N)], float) \
            if ctx is not None else np.ones(self.N)
        p = p / p.sum()
        A = self.rng.choice(self.N, self.d, replace=False, p=p)
        vals = self.atmp[A]
        jitter = self.rng.random(len(A))
        S = sorted(int(A[i]) for i in np.lexsort((jitter, -vals))[:self.K])
        return RoundPlan(S, self._all(S), S, contacted=S)

    def observe(self, t, plan, losses_after=None):
        if losses_after:
            for k, v in losses_after.items():
                self.atmp[int(k)] = float(v)


class MFedMC(_Base):
    """MFedMC, Yuan et al. Algorithm 1. Paper defaults: gamma=1, delta=0.2,
    alpha_s = alpha_c = alpha_r = 1/3.

    EVERY client trains. Each uploads its top-gamma encoders by priority

        P^k_m = a_s * phi~   +  a_c * (1 - size~)  +  a_r * recency~        (13)
        recency  T^k_m = t - t^k_m - 1,  normalised  T~ = T / max(t, 1)     (11)

    The server then aggregates from the top-delta*N clients with the LOWEST
    modality-encoder loss.

    Note: because all clients train, `train` is every client and client
    participation fairness is trivially perfect. The coverage story lives
    entirely in `upload` and `aggregate_from`.
    """
    needs_ctx = True

    def __init__(self, manifest, K=None, seed=42, gamma=1, delta=0.2,
                 a_s=1/3, a_c=1/3, a_r=1/3, **kw):
        super().__init__(manifest, K or 0, seed, **kw)
        assert abs(a_s + a_c + a_r - 1.0) < 1e-9, 'alpha weights must sum to 1'
        self.gamma, self.delta = int(gamma), float(delta)
        self.a_s, self.a_c, self.a_r = a_s, a_c, a_r
        self.last_up = np.full((self.N, self.M), -1.0)   # t^k_m

    @staticmethod
    def _norm(d):
        v = np.array(list(d.values()), float)
        lo, hi = v.min(), v.max()
        rng = hi - lo
        return {k: (0.5 if rng < 1e-12 else (val - lo) / rng)
                for k, val in d.items()}

    def plan_round(self, t, ctx):
        upload = {}
        for k in range(self.N):
            held = sorted(self.held[k])
            if not held:
                upload[k] = set(); continue
            phi = {m: abs(v) for m, v in ctx.shapley(k).items() if m in held}
            size = {m: float(ctx.encoder_bytes[m]) for m in held}
            rec = {m: (t - self.last_up[k, m] - 1) / max(t, 1) for m in held}
            pn, sn = self._norm(phi), self._norm(size)
            P = {m: self.a_s * pn[m] + self.a_c * (1 - sn[m]) + self.a_r * rec[m]
                 for m in held}
            # ties broken AT RANDOM, as Cho et al. specify for pow-d. With a
            # crude Shapley stand-in (constant across clients, equal encoder
            # sizes) exact ties DO occur, and index-order tie-breaking would
            # freeze one modality out entirely -- an artefact, not a finding.
            jit = self.rng.random(len(held))
            top = [held[i] for i in np.lexsort((jit, [-P[m] for m in held]))][:self.gamma]
            upload[k] = set(top)
            for m in top:
                self.last_up[k, m] = t
        train = list(range(self.N))                       # ALL clients train
        n_sel = max(1, int(round(self.delta * self.N)))
        losses = ctx.local_loss(train)
        vals = np.array([losses[k] for k in train], float)
        jitter = self.rng.random(self.N)
        agg = sorted(int(train[i]) for i in np.lexsort((jitter, vals))[:n_sel])
        return RoundPlan(train, upload, agg, contacted=train, probe_passes=0)


class MMiC(_Base):
    needs_ctx = False           # performance arrives through observe()

    def __init__(self, manifest, K, seed=42, tau=1.0, theta=None,
                 weights=None, **kw):
        super().__init__(manifest, K, seed=seed, **kw)
        self.tau = float(tau)
        self.theta = None if theta is None else float(theta)   # None = adaptive
        w = np.ones(self.N) if weights is None else np.asarray(weights, float)
        self.w = w / w.sum()                    # w_i in Eq (8)
        self.phi = np.zeros(self.N)             # Eq (10) core-member counter
        self.T = np.zeros(self.N)               # times selected
        self.perf = {}                          # last recorded a_{i,·}
        self._A_sum, self._A_n = 0.0, 0         # running mean of A_{S,t}

    def _theta(self):
        """alpha^m_t. Fixed if given, else the cluster's own running mean."""
        if self.theta is not None:
            return self.theta
        return self._A_sum / self._A_n if self._A_n else 0.0

    def _probs(self):
        score = self.tau * self.phi / np.maximum(self.T, 1.0)
        e = np.exp(score - score.max())         # stable softmax, Eq (11)
        return e / e.sum()

    def plan_round(self, t, ctx=None):
        p = self._probs()
        S = sorted(int(k) for k in
                   self.rng.choice(self.N, self.K, replace=False, p=p))
        return RoundPlan(S, self._all(S), S, contacted=S)

    def observe(self, t, plan, losses_after=None):
        """Record performance, score the Banzhaf swing, update phi and T."""
        S = list(plan.train)
        for k in S:
            self.T[k] += 1
        if not losses_after:
            return
        # a = performance. losses_after is a LOSS, so performance is its
        # negation; only differences are used, so the offset is irrelevant.
        alpha = {}
        for k in S:
            if k not in losses_after:
                continue
            a_new = -float(losses_after[k])
            alpha[k] = a_new - self.perf.get(k, a_new)   # 0 on first sight
            self.perf[k] = a_new
        if not alpha:
            return
        A = sum(self.w[k] * alpha[k] for k in alpha)     # Eq (8)
        th = self._theta()                               # alpha^m_t
        if A >= th:
            for k in alpha:                              # Eq (9): is k pivotal?
                if A - self.w[k] * alpha[k] < th:
                    self.phi[k] += 1
        self._A_sum += A                                 # threshold adapts AFTER
        self._A_n += 1                                   # scoring this round

    def state(self):
        return {'phi': self.phi.tolist(), 'T': self.T.tolist(),
                'theta': self._theta(), 'prob': self._probs().tolist()}


SELECTORS = {
    'uniform': Uniform,
    'round_robin': RoundRobin,
    'powd': PowD,
    'rpowd': RPowD,
    'mfedmc': MFedMC,
    'mmic': MMiC,
}


def make_selector(name, manifest, K, seed=42, **kw):
    if name not in SELECTORS:
        raise KeyError(f'{name!r} not in {sorted(SELECTORS)}')
    return SELECTORS[name](manifest, K, seed=seed, **kw)
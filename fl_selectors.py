# -*- coding: utf-8 -*-
"""
    from fl_selectors import make_selector, CoverageLog

SOURCES, and what was read
--------------------------
Power-of-Choice   Cho, Wang & Joshi, "Client Selection in Federated Learning:
                  Convergence Analysis and Power-of-Choice Selection
                  Strategies", arXiv:2010.01243. Section 4 and Algorithms 1-2.
MFedMC            Yuan, Han, Wang, Upadhyay & Brinton, "Communication-Efficient
                  Multimodal Federated Learning: Joint Modality and Client
                  Selection", arXiv:2401.16685v2. Algorithm 1, Eqs (8)-(21).

THE ONE STRUCTURAL FACT THAT SHAPES THIS FILE
---------------------------------------------
pow-d and MFedMC do not have the same shape.

  pow-d    probes d candidates, then only m of them TRAIN.
           -> partial participation in the usual sense.

  MFedMC   EVERY client trains every round (Algorithm 1, "Local Learning:
           for each client k in parallel"). The saving comes from each client
           uploading only its top-gamma encoders, and from the server
           aggregating only the top-delta lowest-loss clients.
           -> NOT partial participation. It is an upload filter plus an
              aggregation filter.

So a selector cannot just return "which clients train". Each round is described
by three sets, and different methods constrain different ones:

    RoundPlan.train           clients that run local training
    RoundPlan.upload[k]       which modality encoders client k sends
    RoundPlan.aggregate_from  clients whose uploads enter the average

Encoder m is aggregated in round t  <=>  some k in aggregate_from has
m in upload[k]. That single line is what the whole experiment measures, and it
is invisible to any evaluation that only records which clients participated.

"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

__all__ = ['RoundPlan', 'CoverageLog', 'make_selector', 'SELECTORS']


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
            'GB_up': self.bytes_up / 2**30,
            'GB_down': self.bytes_dn / 2**30,
            'GB_total': (self.bytes_up + self.bytes_dn) / 2**30,
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
            top = sorted(held, key=lambda m: (-P[m], m))[:self.gamma]
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


SELECTORS = {
    'uniform': Uniform,
    'round_robin': RoundRobin,
    'powd': PowD,
    'rpowd': RPowD,
    'mfedmc': MFedMC,
}


def make_selector(name, manifest, K, seed=42, **kw):
    if name not in SELECTORS:
        raise KeyError(f'{name!r} not in {sorted(SELECTORS)}')
    return SELECTORS[name](manifest, K, seed=seed, **kw)

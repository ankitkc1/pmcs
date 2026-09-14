"""Pluggable client-selection policies for federated rounds.
"""
import random
import time

import numpy as np


def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _delta_per_modality(updates, t):
    """delta[m] = clip((t - updates[m]) / max(t, 1), 0.0, 1.0)."""
    denom = max(t, 1)
    return [_clip01((t - u) / denom) for u in updates]


def _g_concave(x):
    """Non-decreasing concave, g(0) = 0: marginal coverage gain shrinks as a
    modality accumulates more contributors in the candidate set."""
    return x / (x + 1.0)


class ClientSelector:
    """Chooses which clients train this round.

    Policies differ in what they need to know. That difference is the point
    of the study, so the interface makes it visible: anything a policy needs
    from outside the server's own bookkeeping must come through `ctx`. A
    policy that never touches `ctx` needs nothing from clients before
    selecting.
    """

    POLICIES = ('uniform', 'poc', 'mics')

    def __init__(self, policy, manifest, K, seed, **kwargs):
        if policy not in self.POLICIES:
            raise ValueError('unknown policy {!r}, expected one of {}'.format(policy, self.POLICIES))
        manifest = np.asarray(manifest)
        if manifest.ndim != 2:
            raise ValueError('manifest must be a 2-D (N, M) array, got shape {}'.format(manifest.shape))

        self.policy = policy
        self.manifest = manifest.astype(np.int64)
        self.N, self.M = self.manifest.shape
        self.K = int(K)
        if not (1 <= self.K <= self.N):
            raise ValueError('K must be between 1 and N={}, got {}'.format(self.N, self.K))
        self.seed = seed

        # The ONE shared stream: `uniform` consumes it every round, in the
        # same order as the pre-refactor code (see select_clients() in
        # train_federated.py, which this must reproduce exactly). `poc` and
        # `mics` touch it only at t==0, so every policy starts from an
        # identical first round.
        self.rng = random.Random(seed)

        # Run-identity config, recorded verbatim in every log row regardless
        # of whether this round's policy happens to use them.
        self.d = int(kwargs.get('d', 2 * self.K))
        self.beta = float(kwargs.get('beta', 0.15))
        self.s_max = int(kwargs.get('s_max', 12))

        # PoC's own stream for weighted-without-replacement candidate
        # sampling. Independent of `self.rng`: PoC is a new policy with no
        # backward-compatibility constraint on its RNG.
        self._poc_rng = np.random.default_rng(seed)

        # Server-side state only. Maintained generically for every policy
        # (observe() is called every round regardless of policy), so every
        # log row carries real updates_m/stale_k numbers -- not just mics'.
        # mics' objective is the only thing that ever reads them back.
        self.updates = [0] * self.M   # rounds in which encoder m was aggregated
        self.stale = [0] * self.N     # rounds since client k was last selected

        self._pending = None
        self._history = []

    # ------------------------------------------------------------ select
    def select(self, t, ctx):
        """Return exactly K distinct client indices, sorted ascending."""
        start = time.perf_counter()
        pre_stale = list(self.stale)
        pre_updates = list(self.updates)
        delta = _delta_per_modality(pre_updates, t)
        round0_fallback = (t == 0)
        forced = []
        forward_passes = 0

        if self.policy == 'uniform':
            selected = self._select_uniform()
        elif self.policy == 'poc':
            if round0_fallback:
                selected = self._select_uniform()
            else:
                selected = self._select_poc(ctx)
                forward_passes = self.d
        elif self.policy == 'mics':
            if round0_fallback:
                selected = self._select_uniform()
            else:
                selected, forced = self._select_mics(t, delta)
        else:
            raise AssertionError(self.policy)

        if len(selected) != self.K or len(set(selected)) != self.K:
            raise AssertionError(
                'select() must return exactly {} distinct client indices, got {}'.format(self.K, selected))
        if any(not (0 <= k < self.N) for k in selected):
            raise AssertionError('select() returned an out-of-range client index: {}'.format(selected))
        selected = sorted(int(k) for k in selected)

        self._pending = {
            'round': t,
            'selected': selected,
            'forced': sorted(int(k) for k in forced),
            'stale_k': pre_stale,
            'updates_m': pre_updates,
            'delta_m': delta,
            'round0_uniform_fallback': round0_fallback,
            'sel_seconds': time.perf_counter() - start,
            'sel_forward_passes': forward_passes,
        }
        return selected

    def _select_uniform(self):
        return sorted(self.rng.sample(range(self.N), self.K))

    def _select_poc(self, ctx):
        n_samples = ctx.n_samples
        weights = np.array([max(0.0, float(n_samples[k])) for k in range(self.N)], dtype=np.float64)
        if weights.sum() <= 0.0:
            weights = np.ones(self.N, dtype=np.float64)
        p = weights / weights.sum()

        d_eff = min(self.d, self.N)
        candidates = self._poc_rng.choice(self.N, size=d_eff, replace=False, p=p)
        candidate_ids = sorted(int(k) for k in candidates)

        losses = ctx.local_loss(candidate_ids)
        ranked = sorted(candidate_ids, key=lambda k: (-losses[k], k))
        return ranked[:self.K]

    def _select_mics(self, t, delta):
        """Never touches ctx: there is no ctx parameter to touch."""
        N, M, K = self.N, self.M, self.K
        stale = self.stale
        manifest = self.manifest
        denom = max(t, 1)

        def cover(S):
            total = 0.0
            for m in range(M):
                c = sum(1 for k in S if manifest[k, m])
                total += delta[m] * _g_concave(c)
            return total / M

        def fair(S):
            return sum(stale[k] / denom for k in S) / K

        def f(S):
            return cover(S) + self.beta * fair(S)

        qualifying = [k for k in range(N) if stale[k] >= self.s_max]
        forced = sorted(qualifying, key=lambda k: (-stale[k], k))[:K]
        S = list(forced)

        while len(S) < K:
            base = f(S)
            best_k, best_gain = None, None
            for k in range(N):
                if k in S:
                    continue
                gain = f(S + [k]) - base
                if best_gain is None or gain > best_gain:
                    best_k, best_gain = k, gain
            S.append(best_k)

        return sorted(S), forced

    # ----------------------------------------------------------- observe
    def observe(self, t, selected, aggregated_modalities):
        """Server tells the selector what actually happened. Call every
        round, for every policy, after aggregation."""
        if self._pending is None or self._pending['round'] != t:
            raise RuntimeError(
                'observe(round={}) called without a matching select() for that round'.format(t))
        self._advance(selected, aggregated_modalities)
        record = self._pending
        record['aggregated_modalities'] = [int(bool(a)) for a in aggregated_modalities]
        self._history.append(record)
        self._pending = None

    def fast_forward(self, t, selected, aggregated_modalities):
        """Resume support: replay a historical round's outcome to bring
        updates[]/stale[] back to where they'd be had training never been
        interrupted, without requiring (or re-running) that round's select()
        and without duplicating it into _history -- that round's log line is
        already on disk. Not part of the select()/observe() contract."""
        self._advance(selected, aggregated_modalities)

    def _advance(self, selected, aggregated_modalities):
        selected_set = set(selected)
        for k in range(self.N):
            self.stale[k] = 0 if k in selected_set else self.stale[k] + 1
        for m in range(self.M):
            if aggregated_modalities[m]:
                self.updates[m] += 1

    # ------------------------------------------------------------- stats
    def stats(self):
        """Per-round history for the log."""
        return {
            'policy': self.policy,
            'seed': self.seed,
            'K': self.K,
            'beta': self.beta,
            's_max': self.s_max,
            'd': self.d,
            'history': [dict(r) for r in self._history],
        }

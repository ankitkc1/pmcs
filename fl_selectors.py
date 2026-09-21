# -*- coding: utf-8 -*-


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
    """MMiC's Banzhaf client selection. Yang et al., "MMiC: Mitigating Modality
    Incompleteness in Clustered Federated Learning", CIKM '25, arXiv:2505.06911.
    Section 4.2 and Eqs (7)-(11).

    THE RULE, from the paper
    ------------------------
        alpha_{i,t} = a_{i,t} - a_{i,t-1}                              Eq (7)
            a is the client's LOCAL MODEL PERFORMANCE (the paper says
            accuracy / F1 / RSum; for segmentation, Dice). It is measured
            AFTER local training, so it costs nothing extra -- unlike pow-d,
            MMiC needs no probe.

        A_{S,t}   = sum_{i in S} w_i alpha_{i,t}                       Eq (8)
        core(i)   = [ A_{S,t} >= theta ] and [ A_{S\{i},t} < theta ]   Eq (9)
            i is PIVOTAL: the cluster clears the bar with i and misses it
            without i. That is the Banzhaf swing.

        phi(i)    = number of rounds in which i was a core member      Eq (10)
        prob(i)   = softmax over i of  tau * phi(i) / T(i)             Eq (11)
            T(i) = how many times i has been selected so far.

    WHY IT MATTERS HERE, AND IT IS THE OPPOSITE OF pow-d
    ----------------------------------------------------
    Section 4.2: "BPI prioritizes clients without missing modalities, thus
    favoring those that contribute more to performance improvements."

    pow-d selects the HIGHEST-loss clients, which in a modality-incomplete
    federation are the modality-POOR sites. MMiC selects the most pivotal
    clients, which are the modality-COMPLETE sites. Same uncontrolled
    variable, opposite sign. Neither reasons about encoder coverage.

    THREE CHOICES THE PAPER LEAVES OPEN — all exposed, all logged
    -------------------------------------------------------------
    theta   Eq (9)'s threshold alpha^m_t is never given a value in the paper.
            It is written with a ROUND subscript and a CLUSTER superscript,
            so it is meant to adapt; the default here is the running mean of
            the cluster's own past returns A_{S,t}.

            This is not a free choice. With theta fixed at 0, Eq (9)'s strict
            inequality makes i core only when dropping it turns the round
            NEGATIVE -- so on any run where clients mostly improve, phi stays
            0 for everyone, Eq (11) stays uniform, and MMiC degenerates
            silently into random selection. That would look like a finding
            and would be an artefact. Pass theta=<float> to pin it, and say
            which you used.
    T(i)=0  Eq (11) divides by it. Guarded with max(T(i), 1), so an
            unselected client scores phi/1 = 0 and every client starts equal
            -- round 0 is therefore uniform, which matches the paper's
            "the standard procedure involves randomly selecting a subset".
    clusters MMiC is clustered FL. With no clustering defined here the whole
            federation is ONE cluster (M=1). Say so in the write-up; it is a
            simplification, not a reimplementation of their clustering.
    """
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


# ===================================================== submodular family
# DivFL      Balakrishnan, Li, Zhou, Himayat, Smith & Bilmes, "Diverse Client
#            Selection for Federated Learning via Submodular Maximization",
#            ICLR 2022. Eqs (3)-(6), Algorithm 1.
# SubTrunc   Kaya et al., "Submodular Maximization Approaches for Equitable
#   UnionFL  Client Selection in Federated Learning", arXiv:2408.13683.
#            Eqs (11)-(14), Algorithm 1.
#
# THE ONE HONEST COMPROMISE, STATED ONCE
# --------------------------------------
# All three score clients by pairwise dissimilarity of their LOCAL GRADIENTS:
#
#     G(S) = sum_{k in [N]} min_{i in S} || grad F_k - grad F_i ||      Eq (3)
#
# Your completed runs never logged gradients, so that matrix cannot be
# reconstructed. Two measured substitutes are provided, and BOTH are reported
# because the choice is not innocent:
#
#   feature='perf'  per-client per-region Dice, from dice_matrix. A
#                   performance-space stand-in: clients that segment
#                   differently are treated as carrying different information.
#   feature='mask'  the client's modality mask. Under modality-specific
#                   aggregation this is arguably the RIGHT space -- a client's
#                   gradient is structurally zero for every encoder it does
#                   not hold, so two clients with disjoint masks cannot have
#                   similar gradients whatever their data.
#
# That DivFL's answer depends on which space you measure diversity in is not
# a limitation of this reimplementation. It is a property of DivFL that no
# paper has had to confront, because in a monolithic model every client's
# gradient lives in the same space. Report both rows.
def _greedy_facility(D, K, extra=None, rng=None, s=None):
    """Maximise the monotone submodular facility-location surrogate

        Gbar(S) = sum_k [ Dmax_k - min_{i in S} D[k, i] ]

    by the greedy algorithm (Eq 5), optionally stochastic over a random
    subset of size s (Eq 6). `extra(k)` adds a per-client modular term, which
    is how SubTrunc's H(S) and UnionFL's -mu*g(S) enter.

    N is 8 here, so full greedy is affordable and is used by default; the
    paper's stochastic variant exists only to avoid scanning large N.
    """
    N = D.shape[0]
    big = D.max(axis=1) if N else np.zeros(0)
    cur = big.copy()                       # min over the empty set -> Dmax
    S = []
    for _ in range(min(K, N)):
        pool = [k for k in range(N) if k not in S]
        if s is not None and rng is not None and s < len(pool):
            pool = list(rng.choice(pool, s, replace=False))
        best, best_k = None, None
        for k in pool:
            gain = float(np.sum(cur - np.minimum(cur, D[:, k])))
            if extra is not None:
                gain += float(extra(k))
            if best is None or gain > best:
                best, best_k = gain, k
        S.append(int(best_k))
        cur = np.minimum(cur, D[:, best_k])
    return sorted(S)


class _Submodular(_Base):
    """Shared plumbing: build the dissimilarity matrix from a feature space."""
    needs_ctx = True

    def __init__(self, manifest, K, seed=42, feature='perf', stale=True,
                 s=None, **kw):
        super().__init__(manifest, K, seed=seed, **kw)
        self.feature = feature
        self.stale = bool(stale)           # DivFL's "no-overheads" variant
        self.s = s
        self._X = None                     # cached feature matrix

    def _features(self, t, ctx):
        if self.feature == 'mask':
            return self.mask.astype(float)
        X = None
        if ctx is not None and hasattr(ctx, 'client_features'):
            X = np.asarray(ctx.client_features(t), float)
        if X is None:
            X = self.mask.astype(float)
        return X

    def _dissim(self, t, ctx):
        """|| x_k - x_i ||_2 over the chosen feature space.

        stale=True reproduces the paper's own evaluation setting: the server
        refreshes the N x N matrix only from clients it has actually heard
        from, so most entries are out of date. It costs no extra
        communication. stale=False refreshes every client every round and is
        charged N probe passes.
        """
        X = self._features(t, ctx)
        if self._X is None or not self.stale:
            self._X = X.copy()
        D = np.linalg.norm(self._X[:, None, :] - self._X[None, :, :], axis=-1)
        # NORMALISE. The papers do not, because their D is built from gradient
        # norms and their lambda / mu were tuned against that scale. Here D is
        # built from a substitute feature space (Dice in [0,1], or a 0/1 mask),
        # whose scale is arbitrary -- so an unnormalised D would silently
        # rescale SUBTRUNC's lambda and UNIONFL's mu and make the published
        # defaults meaningless. Dividing by max(D) makes Gbar's range depend
        # only on N, so lambda and mu carry their published meaning and the
        # 'perf' and 'mask' rows stay comparable to each other.
        mx = float(D.max())
        return D / mx if mx > 0 else D

    def _refresh(self, plan, t, ctx):
        """Stale variant: update only the rows of clients that reported."""
        if not self.stale or self._X is None:
            return
        X = self._features(t, ctx)
        for k in plan.train:
            self._X[k] = X[k]

    def _probes(self):
        return 0 if self.stale else self.N


class DivFL(_Submodular):
    """DivFL. Pure facility location, no second term."""

    def plan_round(self, t, ctx=None):
        D = self._dissim(t, ctx)
        S = _greedy_facility(D, self.K, rng=self.rng, s=self.s)
        plan = RoundPlan(S, self._all(S), S, contacted=S,
                         probe_passes=self._probes())
        self._refresh(plan, t, ctx)
        return plan


class SubTrunc(_Submodular):
    """SUBTRUNC: W(S) = G(S) + lambda * min(b, sum_{i in S} phi(loss_i)).

    Eqs (11)-(13). phi(x) = ln(1 + x) as used in their experiments; defaults
    b = 1.10 and lambda = 0.95, the setting that gave their lowest client
    dissimilarity. lambda = 0 recovers DivFL exactly -- asserted in the tests.

    The truncation min(b, .) is what keeps H(S) submodular. Because it caps
    the loss term, the greedy gain from it decays once the cap is reached,
    which is the mechanism that stops SUBTRUNC collapsing into pure
    loss-greedy selection (i.e. into Power-of-Choice).
    """

    def __init__(self, *a, lam=0.95, b=1.10, **kw):
        super().__init__(*a, **kw)
        self.lam = float(lam)
        self.b = float(b)

    def plan_round(self, t, ctx):
        D = self._dissim(t, ctx)
        loss = np.array([ctx.local_loss([k])[k] for k in range(self.N)], float)
        phi = np.log1p(np.maximum(loss, 0.0))

        def extra_factory(S_state):
            def extra(k):
                before = self.lam * min(self.b, float(phi[S_state].sum())
                                        if len(S_state) else 0.0)
                after = self.lam * min(self.b, float(phi[S_state + [k]].sum()))
                return after - before
            return extra

        S_state = []
        N = self.N
        big = D.max(axis=1)
        cur = big.copy()
        for _ in range(min(self.K, N)):
            ex = extra_factory(S_state)
            best, best_k = None, None
            for k in range(N):
                if k in S_state:
                    continue
                gain = float(np.sum(cur - np.minimum(cur, D[:, k]))) + ex(k)
                if best is None or gain > best:
                    best, best_k = gain, k
            S_state.append(int(best_k))
            cur = np.minimum(cur, D[:, best_k])
        S = sorted(S_state)
        plan = RoundPlan(S, self._all(S), S, contacted=S,
                         probe_passes=max(self._probes(), self.N))
        self._refresh(plan, t, ctx)
        return plan


class UnionFL(_Submodular):
    """UNIONFL: max f_t(S) - mu * g_t(S),  g_t(S) = |(union_{i in u_t} S_i) & S|.

    Eq (14). u_t is a look-back window over previously chosen sets; the paper
    gives {t-5, ..., t-1} as a typical value. mu = 1 in their experiments.

    WHY THIS ONE MATTERS MOST TO YOU
    --------------------------------
    g_t penalises picking anyone chosen in the last `window` rounds. That is
    the same idea as MICS's fairness term beta * (1/K) sum_k s_k(t)/t, in a
    different functional form -- a hard recency penalty rather than a smooth
    staleness reward. So UNIONFL is PRIOR ART for the second half of your
    objective, and the write-up should say so plainly. What remains yours is
    the first half: the per-pool coverage term delta_m(t) * g(|S ∩ P_m|),
    which is defined over modality pools rather than over clients and has no
    counterpart here.

    Unlike DivFL and SUBTRUNC, g_t needs nothing from the clients -- it is
    pure selection history, so it replays exactly.
    """

    def __init__(self, *a, mu=1.0, window=5, **kw):
        super().__init__(*a, **kw)
        self.mu = float(mu)
        self.window = int(window)
        self.history = []

    def plan_round(self, t, ctx=None):
        D = self._dissim(t, ctx)
        recent = set()
        for S_old in self.history[-self.window:]:
            recent |= set(S_old)
        S = _greedy_facility(D, self.K,
                             extra=lambda k: -self.mu * (1.0 if k in recent else 0.0),
                             rng=self.rng, s=self.s)
        self.history.append(list(S))
        plan = RoundPlan(S, self._all(S), S, contacted=S,
                         probe_passes=self._probes())
        self._refresh(plan, t, ctx)
        return plan


SELECTORS = {
    'uniform': Uniform,
    'round_robin': RoundRobin,
    'powd': PowD,
    'rpowd': RPowD,
    'mfedmc': MFedMC,
    'mmic': MMiC,
    'divfl': DivFL,
    'subtrunc': SubTrunc,
    'unionfl': UnionFL,
}


def make_selector(name, manifest, K, seed=42, **kw):
    if name not in SELECTORS:
        raise KeyError(f'{name!r} not in {sorted(SELECTORS)}')
    return SELECTORS[name](manifest, K, seed=seed, **kw)
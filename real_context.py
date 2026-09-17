# -*- coding: utf-8 -*-
"""RealContext — supplies the three quantities the selectors ask for.


COST: use cpow-d, not pow-d
---------------------------
Cho et al. Algorithm 1 gives cpow-d, which replaces the full local loss with a
MINI-BATCH estimate. It is in the paper precisely so you do not pay for a full
pass over d clients every round. With one batch instead of a full epoch the
probe costs roughly 1/50 of what pow-d costs, and the paper treats it as the
practical variant. Use it, and say so.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch


class RealContext:
    """Bridge between your training code and fl_selectors."""

    def __init__(self, server, clients, manifest, n_modalities,
                 probe_batches=1, shapley_every=10, shapley_batches=1,
                 device='cuda'):
        self.server = server            # holds the global encoders + decoder
        self.clients = clients          # list/dict of client handles
        self.mask = np.asarray(manifest)
        self.M = n_modalities
        self.probe_batches = probe_batches
        self.shapley_every = shapley_every
        self.shapley_batches = shapley_batches
        self.device = device
        self.t = 0
        self._shap_cache = {}
        self._probe_passes = 0

        # ---- TODO 1 ---------------------------------------------------
        # number of training cases at each client, for p_k in pow-d
        self.n_samples = {k: len(self.clients[k].train_set)
                          for k in range(len(self.clients))}

        # ---- TODO 2 ---------------------------------------------------
        # bytes per modality encoder, for MFedMC's size term
        self.encoder_bytes = {
            m: sum(p.numel() * p.element_size()
                   for p in self.server.encoder[m].parameters())
            for m in range(self.M)}

    def set_round(self, t):
        self.t = t

    # ------------------------------------------------------------------
    @torch.no_grad()
    def local_loss(self, ids):
        """F_k(w^t): loss of the CURRENT GLOBAL model on client k's own data.

        No optimiser step. No gradient. This is the probe.
        """
        out = {}
        for k in ids:
            c = self.clients[int(k)]

            # ---- TODO 3 -----------------------------------------------
            # load the current global encoders (only those k holds) plus the
            # global decoder into the client's model, WITHOUT touching its
            # local fusion/personalised parts.
            c.load_global(self.server.encoder, self.server.decoder)

            c.model.eval()
            tot, n = 0.0, 0
            for i, batch in enumerate(c.train_loader):
                if i >= self.probe_batches:       # cpow-d: one batch is enough
                    break
                x, y = batch['image'].to(self.device), batch['label'].to(self.device)
                loss = c.criterion(c.model(x), y)
                tot += float(loss) * x.size(0)
                n += x.size(0)
                self._probe_passes += 1
            out[int(k)] = tot / max(n, 1)
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def shapley(self, k):
        """Exact Shapley over the client's own modalities, MFedMC Eq (8).

        With M <= 4 the exact form is 2^M = 16 coalitions, which is cheap on a
        single batch. Recomputed every `shapley_every` rounds and cached --
        the paper recomputes each round, so state this deviation if you use it.
        """
        key = (int(k), self.t // self.shapley_every)
        if key in self._shap_cache:
            return self._shap_cache[key]

        c = self.clients[int(k)]
        held = [m for m in range(self.M) if self.mask[k, m]]
        c.model.eval()

        # value of every coalition: performance using only modalities in Y
        val = {}
        for r in range(len(held) + 1):
            for Y in itertools.combinations(held, r):
                tot, n = 0.0, 0
                for i, batch in enumerate(c.val_loader):
                    if i >= self.shapley_batches:
                        break
                    x = batch['image'].to(self.device)
                    y = batch['label'].to(self.device)

                    # ---- TODO 4 -------------------------------------------
                    # forward using ONLY the encoders in Y. Mask the others the
                    # same way you already handle a missing modality at
                    # inference -- zero the channel, or skip the encoder and
                    # let the decoder see zeros. Use the SAME convention as
                    # your missing-modality path, or the values are not
                    # comparable.
                    pred = c.model.forward_subset(x, set(Y))

                    tot += float(1.0 - c.dice_loss(pred, y)) * x.size(0)
                    n += x.size(0)
                val[frozenset(Y)] = tot / max(n, 1)

        # Shapley value per modality
        Mk = len(held)
        fact = np.math.factorial
        phi = {}
        for m in held:
            rest = [j for j in held if j != m]
            s = 0.0
            for r in range(len(rest) + 1):
                for Y in itertools.combinations(rest, r):
                    w = fact(len(Y)) * fact(Mk - len(Y) - 1) / fact(Mk)
                    s += w * (val[frozenset(Y) | {m}] - val[frozenset(Y)])
            phi[m] = s
        self._shap_cache[key] = phi
        return phi

    @property
    def probe_passes(self):
        return self._probe_passes


# ======================================================================
# The round loop. This replaces your current selection + aggregation.
# ======================================================================
def train_with_selector(policy, manifest, K, seed, T, server, clients,
                        outdir, **sel_kw):
    from fl_selectors import CoverageLog, make_selector

    sel = make_selector(policy, manifest, K, seed=seed, **sel_kw)
    M = manifest.shape[1]
    dec_bytes = sum(p.numel() * p.element_size()
                    for p in server.decoder.parameters())
    enc_bytes = [sum(p.numel() * p.element_size()
                     for p in server.encoder[m].parameters()) for m in range(M)]
    log = CoverageLog(manifest, dec_bytes, enc_bytes)
    ctx = RealContext(server, clients, manifest, M)

    for t in range(T):
        ctx.set_round(t)
        plan = sel.plan_round(t, ctx)

        # 1. local training, only for the clients the plan says
        for k in plan.train:
            clients[k].load_global(server.encoder, server.decoder)
            clients[k].local_train()

        # 2. AGGREGATE PER POOL, and only from clients that UPLOADED m.
        #    This is the line that makes the whole experiment mean anything.
        for m in range(M):
            src = [k for k in plan.aggregate_from if m in plan.upload.get(k, ())]
            if src:
                w = np.array([ctx.n_samples[k] for k in src], float)
                w /= w.sum()
                server.aggregate_encoder(m, [clients[k].encoder[m] for k in src], w)
        dsrc = list(plan.aggregate_from)
        if dsrc:
            w = np.array([ctx.n_samples[k] for k in dsrc], float)
            w /= w.sum()
            server.aggregate_decoder([clients[k].decoder for k in dsrc], w)

        # 3. report losses back (rpow-d needs these; harmless for the others)
        losses = {k: clients[k].last_train_loss for k in plan.train}
        sel.observe(t, plan, losses_after=losses)

        # 4. log
        log.record(t, plan)

        if (t + 1) % 10 == 0:
            print(f'[{policy}] round {t+1}/{T}  '
                  f'encoders updated so far {log.upd.tolist()}', flush=True)

    log.save(f'{outdir}/{policy}_K{K}_seed{seed}.json')
    log.report()
    return log

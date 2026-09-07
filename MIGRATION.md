# Migration: report schema v1 -> v2 (offline re-scoring, corrected accounting)

This release adds an offline re-scoring entrypoint (`evaluate.py`) and a
unified scoring/accounting/reporting stack under `utils/` (`scoring.py`,
`communication.py`, `encoder_coverage.py`, `stats.py`, `report_schema.py`),
and fixes several defects in how Split A and Split B were previously scored
and reported. It does **not** change training, the model, the aggregation
rule, or the optimiser, and it does **not** re-train anything -- Split A and
Split B are re-scored from their existing `model_files/last.pth` checkpoints.

## What changed and why

**1. Post-processing now applies identically to every entity.**
Previously, connected-component filtering was applied only when reporting
the global model, never to the 8 per-client matrices, so client Dice was
understated and client HD95 (42-57mm) was incomparable to the global model's
post-processed HD95 (16-23mm). `utils/scoring.score()` is now the single
entrypoint used for every client and for the residual-zeroed model alike --
there is no branch anywhere in it for "which entity" is being scored, and it
always returns both `dice_raw`/`hd95_raw` and `dice_postproc`/`hd95_postproc`
for every region, for every entity. See `configs/metric_policy.yaml`: WT
uses `largest_cc` (a whole tumour is normally one connected mass); TC and ET
deliberately use `size_filter`, never `largest_cc`, because enhancing tumour
is frequently multifocal or ring-shaped with genuinely disconnected
true-positive components -- `largest_cc` there would delete real tumour, not
noise. The ET-under-500-voxels-becomes-empty heuristic is implemented,
defaulted on, and reported both with and without it (`dice_raw` never has it
applied; `dice_postproc` does), so its effect is visible rather than baked
silently into one number.

**2. Split A and Split B are now scored by the identical, versioned code
path.** Split A was previously scored with an older writer missing
`hd95_policy`, `test_hd95_valid_pairs_only_matrix`, `dice_postproc`,
`hd95_postproc`, and `postproc_policy`, making the two splits incomparable
on HD95. Both `report_v2.json` files in this release are produced by the
same `evaluate.py` invocation of the same `utils/scoring.score()` function
against the same `configs/metric_policy.yaml`, and `report_schema.py`
structurally rejects a report missing any required field before it can ever
be written -- this specific failure mode (fields silently absent) is no
longer possible.

**3. Communication accounting is corrected and now identical between splits
-- READ THIS BEFORE COMPARING TO OLD NUMBERS.**
The historical Split A report counted every client's download as the full
4-modality model (8,406,716 params) regardless of that client's actual
modality mask; Split B's report already used modality-specific accounting.
This was a code difference between the two runs, not a real difference in
what was transmitted. The corrected, single semantics (`utils/
communication.py`, applied identically to upload and download): a client
transmits/receives only the modality encoders in its own mask plus the
shared base (fusion prior + shared decoder parts); the private residual
adapter is never counted, in either direction, for either split. Under this
semantics, **Split A's total communication changes** from its previously
reported 68.39 GB; **Split B's recomputed total is unchanged** at
56,067,340,800 bytes (56.07 GB) -- and, because both splits hold the exact
same multiset of per-modality client counts (just FLAIR/T1ce relabelled),
**Split A's corrected total is also exactly 56,067,340,800 bytes**, not a
new, different number. Communication is also now participation-aware: only
clients actually selected in a given round are billed for that round (both
existing runs are 8/8 full participation every round, so this has no effect
on their totals, but it matters immediately for the upcoming
participation-rate sweep). Every report carries an explicit
`accounting_version` field (`"2.0.0"` in this release) so a pre-fix Split A
number can never be silently averaged or plotted alongside a post-fix
number -- if you have any cached numbers from before this release, discard
the communication totals specifically and re-derive them from
`report_v2.json`.

**4. `test_global_minus_client` is renamed to
`residual_zeroed_minus_personalised` (old key kept as a deprecated alias for
one release).** It was, and remains, exactly the negation of the
personalisation gain -- both compare the same client's own weights with and
without its private residual -- so the old name's implication of an
independent baseline comparison was misleading. The number and its meaning
are unchanged; only the primary key name is new.  Old readers using
`report['test_global_minus_client']` continue to work unmodified in this
release; that alias will be removed in a future one.

**5. `evaluation_partition` is now unambiguous.** Matrix keys carry their
partition in the name (`test_dice_matrix` only appears when
`evaluation_partition == "test"`; `validation_dice_matrix` when
`"validation"`), and `round_selection` is now an explicit
`{"mode": "last_round" | "best_validation", "round_index": N}` block instead
of being inferable only from which round happened to be the final one.

**6. `spacing_mm` is no longer a hardcoded `[1, 1, 1]`.** It is read from
`configs/metric_policy.yaml`'s `expected_spacing_mm`, and, when `--nifti-root`
is supplied to `evaluate.py`, is additionally cross-checked per case against
the real NIfTI affine (asserted within 1e-3, raising on mismatch) rather
than assumed. The BraTS2020 `.npy`-preprocessed pipeline this project trains
on does not itself carry per-case affine metadata forward from the original
NIfTI files, so when `--nifti-root` is not given, the configured value is
used directly and the report's `spacing_source` field for every case says so
explicitly (`"policy_default (no --nifti-root match)"`) -- there is no
remaining code path where `[1, 1, 1]` is silently assumed without being
recorded.

**7. Rounds-to-target is omitted, not fabricated, for single-point
partitions.** The test partition of both existing runs was evaluated at
exactly one round (the final one), so a "rounds-to-target" curve over it
would have every target either trivially satisfied at that one round or
never -- indistinguishable from a real curve unless you already know it's
one point. `should_emit_rounds_to_target()` omits the block entirely for any
partition evaluated at <=1 round; the validation partition (evaluated every
`--eval` rounds throughout training) still gets a real curve.

**8. Per-modality-encoder coverage counters now exist.** `utils/
encoder_coverage.py` tracks `was_updated`/`n_contributors`/`staleness` per
modality per round, derives `idle_fraction`/`mean_staleness`/
`effective_horizon`/`mean_contributors`, and computes a closed-form
`predicted_idle_fraction` (hypergeometric: probability that a uniform sample
of K clients from N contains none of the n_m clients holding modality m) to
check against what's actually observed. At 8/8 full participation (both
existing runs), every modality's idle_fraction is exactly 0.0 by
construction, which is the required self-test baseline before the
participation-rate sweep can be trusted.

## What did NOT change

Training, the model architecture, the FedAvg aggregation rule, the
optimiser, and the private-residual-never-leaves-the-client guarantee are
all untouched. `fusion_decoder.adapter.*_residual` remains excluded from
`global_decoder_prior`, from every upload, from aggregation, and from
broadcast -- `evaluate.py` asserts this exclusion still holds (a non-empty,
logged set of residual tensor names must be found and zeroed when
reconstructing the residual-zeroed comparison model) and raises loudly
rather than silently producing a report where personalisation gain reads as
zero.

## Regenerating the reports

The commands below assume this repo and the completed Split A/B checkpoints
live on the same machine (the training cluster) -- `evaluate.py` needs the
real `model_files/last.pth` and the real BraTS2020 `.npy` volumes, neither
of which are present in every development checkout:

```bash
python evaluate.py \
  --checkpoint runs/splitA/model_files/last.pth \
  --manifest split/generated/data_seed_20260905/splitA/materialized_config.json \
  --data-root /path/to/BraTS2020_npy \
  --partition test \
  --policy configs/metric_policy.yaml \
  --data-seed 20260905 \
  --out runs/splitA/report_v2.json \
  --save-predictions runs/splitA/preds/

python evaluate.py \
  --checkpoint runs/splitB/model_files/last.pth \
  --manifest split/generated/data_seed_20260905/splitB/materialized_config.json \
  --data-root /path/to/BraTS2020_npy \
  --partition test \
  --policy configs/metric_policy.yaml \
  --data-seed 20260905 \
  --out runs/splitB/report_v2.json \
  --save-predictions runs/splitB/preds/ \
  --compare-against runs/splitB/metrics.json --compare-tolerance 1e-4
```

`--compare-against` on the Split B run diffs the newly computed
`test_dice_matrix` against the previously published one and fails (nonzero
exit) if any client/region differs by more than the tolerance -- this is the
acceptance check for this migration; both existing runs' old `metrics.json`
files remain untouched on disk as the comparison baseline.

Each run also writes `per_case_metrics.csv` (per case, per region, per
entity, every raw and post-processed metric from `score()`) next to
`report_v2.json`, so no per-case number is ever collapsed to a mean before
it reaches disk.

`--from-predictions runs/splitA/preds/` re-scores instantly from the masks
saved above under a changed `configs/metric_policy.yaml` (e.g. a different
`min_component_voxels`), with no checkpoint, no data loader, and no forward
pass.

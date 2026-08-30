# pmcs

Server-free federated learning for multimodal brain tumor segmentation with
missing modalities. Each client holds a subset of {FLAIR, T1ce, T1, T2}; there
is no server-held data anywhere in the pipeline.

## Architecture

- **Four modality-specific encoders** (`flair_encoder`, `t1ce_encoder`,
  `t1_encoder`, `t2_encoder`), FedAvg'd independently per round over
  whichever clients hold that modality.
- **A fusion decoder** (region-aware modal fusion + PRM stages) that is
  aggregated in full across all participating clients — no privatised
  filters, no patience/staleness gating.
- **A `FusionAdapter`** at the end of the decoder, conditioned on the
  client's binary modality-availability mask. It holds two parameter sets —
  a `prior` (uploaded and FedAvg'd like the rest of the decoder) and a
  `residual` (trained locally, never uploaded). The effective weight at
  forward time is `prior + residual`, so every client personalises on top of
  the shared prior without exposing that personalisation to the server.
- **Modality-dropout self-distillation** during local training: the encoder
  runs once per batch, and the decoder runs twice — once with all of a
  client's available modalities (teacher) and once with one dropped at
  random (student) — with an MSE loss between the two fused representations
  (teacher detached). Skipped for single-modality clients. Weighted by
  `--lam_sd`.

Per-round aggregation logs update count, staleness, and contributor count for
each modality encoder (see `train_federated.py:log_round_stats`).

## Layout

- `train_federated.py` — federated training loop.
- `infer.py` — evaluate a saved client checkpoint.
- `models/fusion_net.py` — `Encoder`, `Decoder`, `FusionAdapter`,
  `FusionDecoder`, `FusionSegNet`.
- `models/layers.py` — conv blocks, PRM generators, region-aware fusion.
- `dataset/`, `utils/criterions.py`, `utils/predict.py` — data loading,
  losses, and evaluation.
- `split/` — per-client CSV splits for the BRATS layouts used by
  `--setting_options c8` (`18_c8_heter_modalnum`, `20_c8_heter_modalnum`) and
  the smaller `18_c4_c6` layout.

## Usage

1. Download BRATS 2018/2020 (preprocessed to `.npy` volumes) and point
   `--datapath` at it (see `run.sh`).
2. `pip install -r requirements.txt`
3. Train: `bash run.sh` — trains all clients federated over `--c_rounds`
   rounds, evaluating and checkpointing every `--eval` rounds under
   `./results/<--version>/model_files/`.
4. Evaluate a client checkpoint: `bash test.sh` (edit the paths first).

## Acknowledgements

The encoder/decoder conv blocks, region-aware modal fusion, PRM generators,
and the segmentation losses/evaluation build on the implementation from
[FedMEPD](https://github.com/ccarliu/FedMEPD) ("Federated Modality-specific
Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor
Segmentation"). The federated aggregation, personalisation mechanism
(`FusionAdapter`'s prior/residual split), and modality-dropout
self-distillation in this repo are a different design: there is no
server-held dataset, no cross-attention against server-computed anchors, and
no patience-based filter privatisation.

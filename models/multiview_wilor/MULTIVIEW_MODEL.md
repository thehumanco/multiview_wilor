# Multi-view WiLoR main model

The training-time model that sits on top of the multi-session dataloader
([SESSION_DATALOADER.md](../../datasets/SESSION_DATALOADER.md)) and the single-view WiLoR
regressor ([wilor.py](../../../../metric_hand_tracking/wilor/models/wilor.py)). It samples a
variable number of camera views per frame, runs their hand crops through **one** WiLoR, and emits
per-view MANO/camera predictions for a (separately-authored) multi-view loss.

## Files

| File | Purpose |
|---|---|
| [`view_sampling.py`](view_sampling.py) | `select_views()` — per-frame random view selection from a `session_collate` batch. |
| [`multiview_wilor.py`](multiview_wilor.py) | `MultiViewWiLoR` Lightning module: gather → one WiLoR forward → scatter → loss. |
| [`losses.py`](losses.py) | `compute_multiview_loss()` **STUB** (tuple of per-view losses) + the per-view-dict contract. |
| [`test_smoke.py`](test_smoke.py) | 3–5 frame end-to-end smoke test + timing. |

## Architecture / data flow

```
SessionDataModule  ──session_collate──▶  {view_idx: {img,(N,...), mano_params, keypoints, extrinsics, valid, frame_index}}
                                              │
                                    select_views(per frame)        k ~ Uniform{1..MAX_VIEWS}, clamp to #available views
                                              │
                       gather chosen crops across (frame,view) ──▶  one tensor (Σ crops, 3, 256, 256)
                                              │
                              ONE WiLoR.forward_step({'img': ...})   single batched pass, shared weights
                                              │
                        scatter rows back per (frame,view) + attach masked GT + build pred K
                                              │
                       per_view: List[dict]  ──▶  compute_multiview_loss(...)  ──▶  tuple(loss per view)
                                              │
                                  sum → manual_backward → AdamW.step
```

## Key design decisions (all user-confirmed)

- **View→model mapping = random count per frame.** For each source frame we draw
  `k ~ Uniform{1,2,3,4}` (`MAX_VIEWS=4`) and take `n = min(k, #available_views)` views, sampled
  uniformly without replacement. Frames with only 2–3 views (oakink=3, ho3d=5) naturally yield
  1–3; a 73-view interhand frame yields 1–4. Eval is deterministic: `n = min(MAX_VIEWS, A)`, lowest
  view ids. "Available" = views with ≥1 **valid** hand for that frame (placeholder/empty view slots
  from the loader are excluded). Implemented in [`select_views`](view_sampling.py); sampling is
  **per source frame** using each view's `frame_index` + `valid` mask (collate stacks all frames
  per view, so we mask down to one frame at a time).

- **One shared-weight WiLoR, single batched forward pass.** All chosen crops from all
  (frame, view) pairs are concatenated into one tensor and run through a single
  `WiLoR.forward_step` (it only reads `batch['img']`, so a minimal dict suffices). No padding to 4,
  no per-view Python loop over forward passes, no 4 separate weight sets. Outputs are scattered back
  into per-(frame, view) groups by row offsets. This was chosen over (a) 4 independent models and
  (b) pad-to-4 after explicit comparison — it maximizes GPU utilization (one ViT pass) and wastes
  no compute on padded slots.

- **Loss is a stub, owned by someone else.** `compute_multiview_loss(per_view_outputs)` returns a
  tuple of one scalar per selected view. The model only sums them and backprops. The exact keys and
  shapes each per-view dict carries (predictions + masked GT + predicted intrinsics `K` +
  world→cam `extrinsics`) are documented at the top of [`losses.py`](losses.py) as the contract.

- **Predicted camera matrices.** `K` is built per crop from WiLoR's `focal_length` (px at
  `IMAGE_SIZE`) with principal point at the crop center (`IMAGE_SIZE/2`); the world→cam
  `extrinsics [R|t]` come straight from the session `calib.npz` (static, passed through the loader).
  `pred_cam` / `pred_cam_t` (weak-perspective + full-perspective translation) are passed through
  unchanged for the loss author.

- **Training loop: manual optimization, no discriminator.** Mirrors WiLoR's manual-opt pattern
  ([wilor.py:311](../../../../metric_hand_tracking/wilor/models/wilor.py#L311)) — one AdamW over
  the WiLoR params, `manual_backward(sum(losses))`, `optimizer.step()`. The adversarial/discriminator
  branch of WiLoR is intentionally dropped here. Empty batches (no valid hand anywhere) are skipped.

## Wiring into training

`MultiViewWiLoR(wilor_ckpt, wilor_cfg, max_views=4)` loads WiLoR via `load_wilor` (which restores
pretrained weights and fixes the repo-relative MANO paths — **instantiate with the prometheus repo
root as cwd**). Feed it batches from `SessionDataModule` / `session_collate` (`{view_idx: {...}}`).
The runner that ties this together is [`train.py`](../../train.py) (see below).

## Training pipeline

Three files make the model runnable end-to-end:

| File | Role |
|---|---|
| [`train.py`](../../train.py) | argparse orchestrator: build cfg → `SessionDataModule` → `MultiViewWiLoR` → `pl.Trainer.fit`. CLI mirrors `rf-detr/train.py`. |
| [`train.sh`](../../train.sh) | thin wrapper that `cd`s to the **repo root** (required by `load_wilor`'s relative MANO paths) and runs `python -m src.metric_hand_tracking_v2.wilor_v2.train "$@"`. |
| [`configs/train_mix_sessions.yaml`](../../configs/train_mix_sessions.yaml) | **yacs-mergeable** data+train config. |

**Config flow:** `train_mix_sessions.yaml` is the plain-yacs config carrying `DATASETS` +
`GENERAL`/`TRAIN` overrides. `train.py` starts from `default_config()`, pins `MODEL.IMAGE_SIZE=256` and
`DATASETS.CONFIG.EXTREME_CROP_AUG_RATE=0.0` (21-kp hands), merges this yaml, then applies CLI
overrides. `MODEL`/`MANO`/`LOSS_WEIGHTS` come from the pretrained `model_config.yaml` via
`load_wilor` (`model.wilor_cfg`), not from this yaml.

**Run flow:** `default_config()+overrides → SessionDataModule(cfg) + MultiViewWiLoR(ckpt,cfg,...) →
pl.Trainer(..., gradient_clip_val=None) → trainer.fit`. Trainer auto-clip is disabled because the
model uses **manual optimization**; pass `--grad-clip-val > 0` to clip inside `training_step`
instead. (`accumulate_grad_batches` is intentionally not wired — PL ignores it under manual opt.)

**Key CLI knobs** (`python -m src.metric_hand_tracking_v2.wilor_v2.train --help`):
optimization `--epochs/--batch-size/--lr/--weight-decay/--grad-clip-val`; multi-view `--max-views`;
augmentation `--scale-factor/--rot-factor/--rot-aug-rate/--color-scale/--do-flip|--no-flip`
(override `DATASETS.CONFIG.*`); infra `--num-workers/--devices/--precision/--val-check-interval`;
logging `--no-wandb/--project/--log-every-n-steps/--log-media-every-n-steps/--num-log-images`.

**wandb logging** (rf-detr style, all loss-agnostic so it survives whatever the loss becomes):
- scalars: `train/loss`+`val/loss`, `{train,val}/num_views`, `/num_hands`, `/avg_hands_per_view`,
  `train/grad_norm` (when clipping), and `lr-*` via `LearningRateMonitor`. `train/*` log on step
  and epoch; `val/*` on epoch.
- media: `{train,val}/predictions` — predicted (red) vs GT (green) 2D keypoints overlaid on
  un-normalized crops, every `--log-media-every-n-steps` train steps and on the first val batch
  per epoch. Guarded on `isinstance(self.logger, WandbLogger)`, so non-wandb runs skip it.
- `WANDB_MODE=offline` works for dry checks. `--no-wandb` disables wandb entirely.

**Loss is still a STUB.** `compute_multiview_loss` returns zeros, so `train/loss`/`val/loss` log as
**0** until someone implements it — expected, not a bug. [`losses.py`](losses.py) carries the
reference scaffolding (reusable `Keypoint2D/3DLoss`/`ParameterLoss` via `model.wilor.*`, `aa_to_rotmat`,
`LOSS_WEIGHTS` via `model.wilor_cfg`, and commented-out per-view + multi-view-consistency sketches).

**Note on `_split_outputs`:** each per-view dict now also carries the masked crop `img`
(`batch[view]["img"][m]`) purely so media logging has pixels; it is not consumed by the loss.

This package no longer depends on the upstream HaMeR repo: `get_example`/`expand_to_aspect_ratio`
and `default_config` are imported from the single-view WiLoR tree
(`src/metric_hand_tracking/wilor/`), and the trivial HaMeR `Dataset` base was dropped
(`SessionDataset` is now a plain class).

## Smoke test results (2026-06-10, GH200)

Run from the repo root. `wilor_v2/datasets/__init__.py` is webdataset-free, so the test imports
the session loaders normally:

```
conda activate prometheus
OPENBLAS_NUM_THREADS=1 python \
  -m src.metric_hand_tracking_v2.wilor_v2.models.multiview_wilor.test_smoke
```

On one oakink session (2610 frames, 3 views), `batch_size=4`:
- collate → `{0,1,2}` view dicts, 8 valid hands each.
- `select_views` produced per-frame view counts `{2, 3, 3, 2}` — all in `[1, min(4,3)]`. ✓
- single batched forward: 20–24 crops; **cold 354 ms, warm 187 ms** on CUDA.
- per-view shapes verified: `global_orient (N,1,3,3)`, `hand_pose (N,15,3,3)`, `betas (N,10)`,
  `K (N,3,3)`, `pred_cam_t (N,3)`, `gt_keypoints_3d (N,21,4)`.
- loss stub returned 10 per-view scalars; `backward()` reaches the model. ✓  → **PASS**.

## Observed optimizations / notes

- The single-pass gather/scatter keeps GPU work to one ViT forward; index math is cheap CPU integer
  ops. No per-hand or per-view Python loop on the hot path.
- `img_size` is `[W,H]`; if a future loss reprojects through the **full-image** K (not the crop K),
  it must undo the `get_example` crop warp (`_trans` is available on the per-hand sample but not
  currently surfaced to the per-view dict — add it to `_split_outputs` if needed).
- Optional `cfg.MAX_CROPS_PER_STEP` cap (not yet wired) would bound memory when a batch has many
  hands × views (e.g. interhand). Add if OOM appears at scale.
- `EXTREME_CROP_AUG_RATE` **must stay 0** for hands (21 kp) — the smoke test sets it explicitly.

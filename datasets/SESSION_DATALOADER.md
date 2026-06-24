# Multi-session WiLoR-v2 dataloader

Training dataloader that consumes our **own multi-view / multi-hand intermediate format** and
feeds the multi-view WiLoR-v2 model. Mixes any number of sessions by configured proportions
(which must sum to 1).

## Files

| File | Purpose |
|---|---|
| [`session_dataset.py`](session_dataset.py) | `SessionDataset` — map-style dataset over one `session_xxx/`. World→camera conversion of MANO/joints, per-`(hand, view)` sample emission. Reuses `get_example` from the single-view WiLoR tree. |
| [`mixed_session_dataset.py`](mixed_session_dataset.py) | `session_collate` (batches variable-hand view-groups), `MixedSessionDataset` (proportion-weighted `ConcatDataset` + `WeightedRandomSampler`), `SessionDataModule` (Lightning datamodule for session-format datasets). |
| [`../configs/train_mix_sessions.yaml`](../configs/train_mix_sessions.yaml) | Inline **dataset-root** paths + proportions plus training overrides. |
| [`data_processing/`](data_processing/) | Dataset download/conversion helpers that produce this session format. |

## Config: dataset roots + proportions

Each `DATASETS.TRAIN` / `DATASETS.VAL` entry is one **dataset**: `PATH` points at a root dir
containing many sessions, `WEIGHT` is that dataset's overall sampling proportion. Sessions are
auto-discovered by `discover_sessions()` (any descendant dir containing `calib.npz`), so the
same entry works for flat layouts (`oakink/oakink_sessions/<scene>/`) and split layouts
(`ho3d/ho3d_sessions/train/<seq>/`, `interhand_sessions/train/<capture>/`). WEIGHTs must sum to 1.
A dataset's proportion is spread uniformly over its frames (so larger sessions contribute more).

Registered in [`__init__.py`](__init__.py) so `SessionDataset`/`SessionDataModule` are importable
without importing any upstream HaMeR modules.

## On-disk session format

```
session_xxx/
  calib.npz                 # per-view pinhole params (static); view 0 == world frame
  frames/000001.npz         # one logical sample == one multi-view, multi-hand frame
  images/view0/000001.jpg
  images/view1/000001.jpg
  ...
```

**`calib.npz`** — arrays indexed by view `V`:
`K (V,3,3)`, `dist (V,D)`, `R_world_to_cam (V,3,3)`, `t_world_to_cam (V,3)`, `img_size (V,2)` `[W,H]`.
View 0 is the world frame (R=I, t=0).

**`frames/NNNNNN.npz`** — a single pickled `data` key holding
`{frame_id: int, is_annotated: bool, hands: list}`, where each hand is:
```python
{
  "hand_id": int,
  "side": "right" | "left",
  "mano": {"global_orient": (3,) axis-angle WORLD frame, "hand_pose": (45,), "betas": (10,)},
  "views": { view_idx: {"bbox": (4,) [x1,y1,x2,y2], "joints_2d": (21,2), "joints_3d": (21,3),
                        # optional per-view "K": (3,3), "trans": (3,)},
             ... only views that actually see this hand ... }
}
```

## Output format

`SessionDataset[idx]` returns **one frame** = a list of `NUM_VIEWS` view-groups; each view-group
is a list of per-hand dicts (one per hand that view sees). Empty view slots get a single
`valid=False` placeholder.

Each per-hand dict mirrors the field schema of
[`vitdet_dataset.py`](../../../../metric_hand_tracking/wilor/datasets/vitdet_dataset.py)
(`img, personid, box_center, box_size, img_size, right`) **plus** training GT:

| Field | Shape / type | Notes |
|---|---|---|
| `img` | `(3, IMG_SIZE, IMG_SIZE)` | normalized crop; left hands flipped (by `get_example`) |
| `personid` / `hand_id` | int | hand index in the frame |
| `box_center` | `(2,)` | bbox center, full-image px |
| `box_size` | float | square box after `expand_to_aspect_ratio` |
| `img_size` | `(2,)` | full source image `[W, H]` |
| `right` | float | 1.0 right / 0.0 left |
| `mano_params` | `{global_orient (3), hand_pose (45), betas (10)}` | **camera frame**, axis-angle |
| `betas` | `(10,)` | |
| `keypoints_3d` | `(21,4)` | **camera frame**, +conf |
| `keypoints_2d` | `(21,3)` | crop frame, +conf |
| `extrinsics` | `(3,4)` | `[R\|t]` world→cam for the view |
| `valid` | bool | False for padded/empty view slots |

`session_collate` concatenates all hands across the batch per view, returning
`{view_idx: {field: tensor (N_hands, ...), 'valid': bool mask, 'frame_index': which batch frame}}`.

## Key design decisions

- **Crop via `get_example` (with augmentation)**, not the inference `ViTDetDataset` crop:
  `get_example` crops AND transforms `joints_2d`/`joints_3d`/`global_orient` by the same warp, so
  GT stays aligned under scale/rot/flip/color aug. We only mirror `ViTDetDataset`'s *field schema*.
  - Pros: one call keeps the RGB crop, 2D keypoints, 3D keypoints, MANO global orientation, and
    left-hand flip convention in the same augmented coordinate system; training sees the same
    scale/rotation/color jitter path as the single-view WiLoR datasets; no duplicate augmentation
    math is needed in `SessionDataset`; and the returned fields still match what downstream WiLoR
    code expects from `ViTDetDataset`.
  - Cons: this couples the session loader to training-time WiLoR augmentation semantics rather than
    the simpler inference preprocessor; some `get_example` options are body-dataset assumptions
    (`EXTREME_CROP_AUG_RATE` must stay 0 for 21 hand joints); `_trans` must be carried if later code
    needs to map crop-space predictions back to image pixels; and train/eval preprocessing can
    diverge subtly from `ViTDetDataset` defaults unless config values such as bbox scaling,
    antialiasing, flip handling, mean/std, and `train` mode are kept intentional.
- **World→camera global_orient**: `R_cam = R_world_to_cam[v] @ exp(global_orient_world)`, returned
  as axis-angle (`scipy Rotation`). Done **before** `get_example`, which then applies the crop's
  flip/rot on top.
- **`joints_3d`**: assumed already per-view camera frame. Set `DATASETS.JOINTS3D_IN_WORLD: True`
  in the config to transform `X_cam = R @ X_world + t` instead.
- **Empty-frame filtering**: `DATASETS.FILTER_EMPTY_FRAMES` (default **True**) indexes only
  frames with ≥1 annotated hand, so we never sample an all-padding frame (e.g. dexycb's
  unannotated leading frames). One-shot scan at `SessionDataset.__init__` reading just the small
  `data` dict (not images). Set False to keep every frame. The kept-frame list is cached to
  `<session>/.kept_frames_cache.json` (`DATASETS.CACHE_FRAME_INDEX`, default True) so the scan
  runs once per session — fingerprinted on the frame set so it rebuilds if frames change, and
  silently skipped on read-only session dirs.
- **Proportional mixing**: per-sample weight = `session_weight / len(session)`, so each session
  contributes its configured proportion of draws regardless of raw frame count (map-style
  analogue of `wds.RandomMix`). WEIGHTs are validated to sum to 1.0 at load.

## Gotchas

- **View count varies by dataset**: `num_views` comes from `calib.npz`, so a frame yields that
  many view-slots — oakink=3, ho3d=5, **interhand=73** (multi-camera rig). The loader emits all
  available views; selecting the subset passed through MultiViewWiLoR is a training-loop decision
  downstream of the collate, not the loader's job. If you need a fixed cap, subsample views in the
  training step or add a
  `MAX_VIEWS`/view-selection option to `SessionDataset.__getitem__`.
- `DATASETS.CONFIG.EXTREME_CROP_AUG_RATE` **must be 0** for hands — extreme cropping in
  `get_example` indexes body keypoints (>21) and will `IndexError` on 21 hand joints. The WiLoR
  hand configs already disable it; keep it off here.

## Open verification points (confirm against a real generated session)

1. Whether `frames` `joints_3d` is world or per-view camera frame (toggle `JOINTS3D_IN_WORLD`).
2. Whether per-view `K`/`trans` inside `frames` should override `calib.npz` `K` (currently
   `calib.npz` `K` is used; per-view override not yet wired).
3. Exact axis-angle convention the MANO head expects after the world→cam compose — validate by
   reprojecting `keypoints_3d` (camera frame) through `K` onto the crop.

## Usage

```python
from src.metric_hand_tracking.wilor.configs import default_config
from src.metric_hand_tracking_v2.wilor_v2.datasets import SessionDataModule

cfg = default_config()           # + merge train_mix_sessions.yaml / experiment cfg
dm = SessionDataModule(cfg)
dm.setup()
for batch in dm.train_dataloader():
    # batch[view] -> dict of stacked per-hand tensors for that view
    ...
```

Wire into training by merging `wilor_v2/configs/train_mix_sessions.yaml` and using
`SessionDataModule`. `get_example`/`expand_to_aspect_ratio` and `default_config` come from the
single-view WiLoR tree (`src/metric_hand_tracking/wilor/`); the loader no longer depends on the
upstream HaMeR repo.

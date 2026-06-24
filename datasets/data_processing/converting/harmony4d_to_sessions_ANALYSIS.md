# Harmony4D → Session Format: Feasibility Analysis

**Status:** Conversion **blocked** — Harmony4D does not contain hand pose data. This
document records what was investigated, why a straight conversion is not possible,
and the realistic options for moving forward. No converter was written.

**Author note:** Written 2026-06-09 after exploring the source data and the existing
`ho3d_to_sessions.py` converter.

---

## 1. Goal

Convert Harmony4D into the project's hand-centric multi-view "session" format
(same layout the HO-3D converter produces), located next to the source data.

### Target format (recap)

```
session_xxx/
    calib.npz                 # per-view pinhole params (view0 = world frame)
    frames/000001.npz         # one logical training sample = one file
    images/view0/000001.jpg
    images/view1/000001.jpg
```

`calib.npz`: `{ K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3), t_world_to_cam:(V,3), img_size:(V,2) }`

`frames/NNNNNN.npz` → object array `"data"`:
```python
{ "frame_id": int, "is_annotated": bool,
  "hands": [ { "hand_id": int, "side": "left"|"right",
               "mano": {"global_orient":(3,), "hand_pose":(45,), "betas":(10,)},  # world frame
               "views": { k: {"bbox":(4,), "joints_2d":(21,2), "joints_3d":(21,3)} } } ] }
```

Reference implementation that produces this format:
`src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/ho3d_to_sessions.py`.

---

## 2. What Harmony4D actually contains

Source: `/lambda/nfs/hfm/qasim/hand_kp_dataset/harmony4d`
Inspected sequence: `train/01_hugging/001_hugging` (301 frames; verified pattern across
15 train + 7 test sequences).

```
<split>/<action>/<sequence>/
    ego/aria{01,02}/images/{rgb,left,right,rotated_*}/NNNNN.jpg   # 2 Aria, fisheye
    exo/cam{01..22}/images/NNNNN.jpg                             # 22 cams, 3840x2160
    colmap/workplace/{cameras.txt, images.txt, points3D.txt, *.pkl}
    processed_data/
        smpl/NNNNN.npy        # dict {aria01, aria02} → full-body SMPL
        poses3d/NNNNN.npy     # dict {aria01, aria02} → (17,4) body keypoints
        poses2d/<cam>/NNNNN.npy  # (45,2) projected body+landmark joints
        bbox/<cam>/NNNNN.npy     # (4,) int  [x1,y1,x2,y2]
```

### SMPL payload (per person)
```
global_orient (3,)   transl (3,)   body_pose (69,)=23 joints   betas (10,)
vertices (6890,3)    joints (45,3)
```

### Calibration
- `cameras.txt`: 24 cameras, all `OPENCV_FISHEYE` (params: fx fy cx cy k1 k2 k3 k4).
  IDs 1–2 = Aria ego (1408×1408); IDs 3–24 = exo cam01..cam22 (3840×2160).
- `images.txt`: per-image quaternion (wxyz) + translation in COLMAP frame.
- `aria_from_colmap_transforms.pkl` / `colmap_from_aria_transforms.pkl`: 4×4 SE(3)
  per person, COLMAP↔Aria.

---

## 3. Why conversion is blocked

The target is **hand-centric**; Harmony4D is **body-centric** with **no hand pose**.

| Target field | Needs | Harmony4D has | Gap |
|---|---|---|---|
| `mano.hand_pose` | 15 articulated finger joints | vanilla SMPL body — **0 finger DOF** | not present |
| `mano.global_orient` | hand root orientation | **body** root orientation only | wrong entity |
| `mano.betas` | MANO **hand** shape | SMPL **body** shape | wrong entity |
| `joints_3d (21,3)` | 21 hand keypoints | 17 body kps; or 45 body+landmark | hands not articulated |
| `joints_2d (21,2)` | 21 hand keypoints projected | (45,2) body+landmark | same |

Detail on the 45-joint set: indices 0–22 are body skeleton, 23–33 are face/foot
landmarks, and the handful of "hand" entries (e.g. ~35–39 left, 40–44 right) are
**coarse regressed surface points**, not the wrist + 5×4 finger joints MANO requires
(only ~6 hand-region points per hand exist vs. 21 needed). Vanilla SMPL has no fingers,
so finger articulation **cannot be derived** — recovering it is an *estimation* problem,
not a format conversion.

**Consequence of a "faithful" conversion:** every hand field would be zero-filled with
`is_annotated=False`. Images + calibration carry over, but the supervision target (hand
pose) is empty — the result cannot train a hand model. Not worth doing.

---

## 4. Options

### Option A — Stop / use a hand-native dataset *(simplest; recommended if true GT needed)*
Harmony4D is not suitable for the hand-centric target. If real hand ground truth is
required, use a dataset that ships MANO / 21-joint hand keypoints. HO-3D already has a
working converter (`ho3d_to_sessions.py`) producing exactly this format.

- **Effort:** none (reuse existing).
- **Output:** true hand GT.

### Option B — Pseudo-label with a hand estimator *(large build; pseudo-GT only)*
Treat Harmony4D as raw multi-view video and *generate* hand labels:

1. **Locate hands** per frame/view using COLMAP extrinsics + SMPL wrist (joints idx
   20=left, 21=right) projected into each of the 22 exo + 2 ego views; or body bbox.
2. **Crop** each hand region per view.
3. **Estimate MANO** per crop with the project's WiLoR/hand-pose model
   (`src/metric_hand_tracking_v2/...`, `src/metric_hand_tracking/wilor`).
4. **Multi-view fuse / triangulate** per-view estimates into a single world-frame MANO
   + 21 joints_3d; reproject to each view for joints_2d/bbox.
5. **Write** session format (mirror `ho3d_to_sessions.py`: `build_calib`,
   `convert_frame`, `write_image`, resume logic, nested tqdm).

- **Effort:** substantial — a detect→crop→estimate→fuse pipeline, plus COLMAP
  fisheye→pinhole handling and world-frame rebasing (view0 = world).
- **Output:** **pseudo-GT** — noisy, model-dependent, not ground truth.
- **Caveats:** fisheye distortion (OPENCV_FISHEYE) on all 24 cams; COLMAP frame must be
  rebased so view0 is world; exo↔ego scale/registration via the aria/colmap transforms.

### Option C — Geometry-only conversion *(not recommended)*
Convert calib + images + existing wrist/body-bbox into the layout, hand pose zero-filled,
`is_annotated=False`.

- **Effort:** moderate.
- **Output:** **no hand labels** — a multi-view image+calib dump only. Listed for
  completeness; produces nothing trainable for hands.

---

## 5. Recommendation

- If you need **true hand GT** → **Option A** (use HO-3D or another hand-native dataset).
- If you specifically need Harmony4D scenes and can accept **pseudo-labels** → **Option B**,
  scoped as its own estimation pipeline (not a thin converter). Confirm pseudo-GT is
  acceptable before building.
- **Avoid Option C** unless you only want the multi-view images/calibration.

---

## 6. Reuse pointers (if Option B proceeds)

- **Format writer pattern:** `src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/ho3d_to_sessions.py`
  — `build_calib` (calib.npz, view0=world rebasing), `convert_frame` (frame dict),
  `write_image` (symlink/copy), `session_is_complete` (resume), nested-tqdm progress.
- **Pinhole/weak-persp cam conversion:** `src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/run_whim_to_wilor_cam.py`.
- **Hand estimator + MANO:** `src/metric_hand_tracking_v2/` (dataset loaders, camera math,
  data_classes with `.save()/.load()`), `src/metric_hand_tracking/wilor`,
  `src/utils/hand_utils.py` (FrameHandDetection/Pose dataclasses).
- **Coordinate conventions:** OpenGL→OpenCV `FLIP=diag(1,-1,-1)`; world→cam via
  `inv(M_0)@M_k`; MANO global_orient rebased to world (`R_cam_to_world @ R_global_cv`).
```

# OakInk-v2 → multi-view "session" format conversion

Script: [`oakink_to_sessions.py`](oakink_to_sessions.py)

Converts the OakInk-v2 checkout
(`/lambda/nfs/hfm/qasim/hand_kp_dataset/oakink`) into the project's multi-view
"session" format, one session per sequence, written by default to
`<oakink_root>/oakink_sessions/`.

## Source schema (OakInk-v2 `anno_preview/<seq>.pkl`)

Each sequence pickle is a dict; the keys we use:

- `cam_def`: `{cam_id: name}` for the 4 cameras
  (`104422071041=allocentric_top`, `104422070088=allocentric_left`,
  `104422070904=allocentric_right`, `104422070969=egocentric`).
- `frame_id_list`: sparse image frame ids (1, 5, 9, …). **An image `frame_id`
  indexes `raw_mano` / `cam_intr` / `cam_extr` directly** — it equals the mocap
  index, so there is no separate frame→mocap remap to do.
- `cam_intr[name][fid]`: `(3,3)` K. **Static across frames** (verified max-diff 0).
- `cam_extr[name][fid]`: `(4,4)` **world→cam**. Static for the 3 allocentric
  cameras; **NOT static for egocentric** (head-mounted, moves up to ~0.2/frame).
- `raw_mano[fid]`: `{rh,lh}__pose_coeffs (1,16,4)` quaternion (wxyz; joint 0 =
  global_orient, joints 1..15 = hand pose), `{rh,lh}__tsl (1,3)` wrist world
  translation, `{rh,lh}__betas (1,10)` shape. `global_orient` is already in the
  world frame.

Extracted images: `data/extracted/<seq>/<cam_id>/NNNNNN.png` (848×480 RGB PNG).

`raw_smplx`, `obj_list`, `obj_transf` (full body + objects) are **out of scope**.

## Output format (target "session" format)

```
<out>/<session>/
  calib.npz   { K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3),
                t_world_to_cam:(V,3), img_size:(V,2) }   # V=3, view0 = world frame
  frames/NNNNNN.npz   # np.savez(..., data=np.array(data_dict, dtype=object))
  images/view0/NNNNNN.jpg  view1/...  view2/...
```

`frames/NNNNNN.npz` `data` dict:
```python
{ "frame_id": int, "is_annotated": True,
  "hands": [ { "hand_id": 0|1, "side": "right"|"left",
               "mano": {"global_orient":(3,), "hand_pose":(45,), "betas":(10,)},  # world frame, axis-angle
               # only the views that actually see this hand:
               "views": { k: {"bbox":(4,), "joints_2d":(21,2), "joints_3d":(21,3)} } } ] }
```
`joints_3d` is in that view's OpenCV camera frame; `bbox` is xyxy with 15% margin.

## Key conventions / geometry

- **3 views only.** The egocentric camera is dropped because its extrinsic is
  non-static and a single `calib.npz` cannot represent a moving camera. View
  order is fixed: `view0=allocentric_top` (world), `view1=allocentric_left`,
  `view2=allocentric_right`.
- **view0 = world frame.** OakInk extrinsics are world→cam in OakInk's own world
  frame. We rebase so the new world == camera-0 frame:
  - `E_k` = OakInk world→cam_k (static, from the first frame).
  - Calib stores `world→cam_k (rebased) = E_k @ inv(E_0)` → view0 gets
    `R=I, t=0`.
  - MANO world joints (in OakInk world) are mapped into the new world by
    `p' = E_0 @ [p;1]` before projecting. (Subtle: because the new world IS the
    cam0 frame, the forward map is `E_0`, not its inverse.)
- **MANO → 21 joints.** Quaternion coeffs → axis-angle via
  `scipy Rotation.from_quat(q[:,[1,2,3,0]]).as_rotvec()`; run `smplx` MANO
  (`use_pca=False, flat_hand_mean=True`, per-side `is_rhand`) to get 16 joints +
  778 verts; append the 5 fingertip vertices
  `{thumb:744,index:320,middle:443,ring:554,pinky:671}`. Do **not** reorder to
  OpenPose: the target HO-3D session layout is MANO's 16 joints followed by
  thumb/index/middle/ring/pinky fingertips. World joints = `joints21 -
  joints21[0] + tsl`.
- **MANO hand pose order.** `mano.hand_pose` is `raw_mano` joints 1..15 converted
  from quaternion to axis-angle and flattened without remapping. This is native
  MANO pose-vector order, matching HO-3D `handPose[3:]`:
  `[index1,index2,index3,middle1,middle2,middle3,pinky1,pinky2,pinky3,
  ring1,ring2,ring3,thumb1,thumb2,thumb3]`. The 21-joint arrays above are the
  fields that use the HO-3D session keypoint order.
- **Both hands** are emitted: `hand_id=0` right, `hand_id=1` left.
- **Per-hand view visibility:** a view is included for a hand only if all joints
  have `z>0` and ≥50% of joints fall inside the image. A hand with no visible
  view is dropped from that frame.
- **Images** are transcoded PNG→JPG (`cv2.imread`+`imwrite`, quality 95). No
  symlinking (would leave `.png` names).
- **MANO weights:** `src/metric_hand_tracking/wilor/mano_data/MANO_{LEFT,RIGHT}.pkl`.
  `smplx.create` wants a `mano/MANO_*.pkl` subdir layout; the script builds a
  temp symlink dir automatically.

## Why a standalone script (not extending `dataset.py`)

`src/metric_hand_tracking_v2/utils/dataset.py` is a **loader** for a different
(stereo, YAML-calib) on-disk layout — it does not read this session format and
has nothing to extend for conversion. The converter is self-contained; the only
reuse is the canonical 21-joint construction from the MANO wrapper.

## Verification performed (single session, smoke)

- Multi-view geometry: per hand, `joints_3d` back-projected from each view to
  world agree to **0.000 mm** spread (views are mutually consistent).
- Reprojection overlay onto the real PNGs: right/left keypoints land cleanly on
  both hands in all 3 views, with tight bboxes (confirms quaternion→aa, MANO
  forward, 21-joint order, and the view0-rebase extrinsics).
- On-disk schema matches the target exactly: `calib.npz` keys/shapes
  `(3,3,3)/(3,5)/(3,3,3)/(3,3)/(3,2)`, view0 R=I/t=0; `frames/*.npz` `data`
  dict with both hands and per-view `bbox(4)/joints_2d(21,2)/joints_3d(21,3)`.
- `--verify` (in-bounds + z>0 assertions over a few frames) passes.
- Throughput: ~2.7 min/session on CPU (dominated by PNG→JPG transcode).

## Usage

Smoke test (first session, with assertions, to a scratch dir):
```bash
conda activate prometheus
python \
  src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/oakink_to_sessions.py \
  --limit 1 --verify --out_root /tmp/oakink_sessions_smoke
```

Full conversion (all 627 sequences → `<oakink_root>/oakink_sessions/`,
resumable — re-running skips completed sessions), parallelized over 16 processes:
```bash
conda activate prometheus
python \
  src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/oakink_to_sessions.py --workers 16
```

### Parallelism
- The job is embarrassingly parallel **across sessions** (each session is
  independent: own pickle, own output dir). `--workers N` runs a
  `multiprocessing` Pool (spawn context, safe with torch) over sessions via
  `imap_unordered`; each worker builds its MANO models once and converts whole
  sessions. Resume/skip is per-session so the pool is restart-safe.
- Workers are **CPU-only** for MANO (many processes can't share one GPU
  efficiently, and CUDA+fork is unsafe); `torch.set_num_threads(1)` per worker
  avoids BLAS oversubscription. `--device cuda` only applies to a serial
  (`--workers 1`) run.
- The bottleneck is PNG→JPG transcode + npz writes (I/O + libjpeg), which scale
  well with processes. Serial is ~2.5 min/session (~43 h for 627); ~16 workers
  brings it to roughly ~3 h, I/O permitting. Pick `--workers` near your core
  count; inner per-frame progress bars are suppressed in workers (only the
  session-level bar shows).

## Changelog (deviations from the original plan)

- **Dropped the egocentric view (4 → 3 views).** The plan assumed all 4 cameras
  were static; inspection during implementation showed the egocentric extrinsic
  varies frame-to-frame (head-mounted), which a static `calib.npz` cannot
  represent. The 3 allocentric views remain a strong static multi-view rig.
- **view0-rebase fix.** Initial implementation rebased MANO world points by
  `inv(E_0)`; the correct map is `E_0` (the new world frame *is* the cam0
  frame). Caught by the reprojection check (0% in-bounds → 100% after fix).

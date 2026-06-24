# Convert HO-3D v3 → multi-view session format

## Summary (completed)

**Date:** 2026-06-09. **Output:** `/lambda/nfs/hfm/qasim/hand_kp_dataset/ho3d/ho3d_sessions/{train,evaluation}/`
(images are **symlinks** — see storage note below). Script:
[ho3d_to_sessions.py](ho3d_to_sessions.py).

**Run:** `python ho3d_to_sessions.py --split all --verify --symlink-images --out_root .../ho3d_sessions`

**Produced:**
- **train** — 24 sessions, 32,132 frames (30,312 annotated), 55 views.
  Multi-cam (5 views unless noted): ABF1, BB1, GPMF1, GSF1, MDF1, SiBF1; ShSu1 (4),
  SB1 (3), SMu4 (3). Single-cam (1 view): MC1/2/4/5/6, ND2, SM2/3/4/5, SMu1,
  SS1/2/3, SiS1.
- **evaluation** — 4 sessions, 5,812 frames, 13 views: AP1 (5), MPM1 (5), SB1 (2,
  partial group — only cams 1&3 present in eval), SM1 (1).

**Verification:** all sessions pass the multi-view invariant — the wrist/root joint
maps to one world point from every view to **< 1 mm** (asserted in `verify_session`,
both splits). Per-view reprojection of stored joints matches projected `joints_2d`.
Format validated programmatically on every session (calib shapes, frame/hand/mano
shapes, view0 = identity extrinsics, image presence): **0 errors** across all 28
sessions. Visual overlay of `joints_2d` on `images/view0` lands on the hand.

**Key decisions / deviations from the original plan:**
- **Split handling.** No on-disk val split exists; per request, **eval is used as
  the test set** (no dedicated val). Added `--split all` to do both in one run.
- **Eval pose is WITHHELD by the benchmark** (README): eval pkls have no
  `handPose`/`handBeta`, and `handJoints3D` is only the **(3,) root joint** plus a
  `handBoundingBox`. So eval frames are emitted with `is_annotated=False`, per-view
  `bbox` + root joint (stored in `joints_3d[0]`; rest zero), and **mano zero-filled**
  — so withheld pose can never be used as supervision, while bbox/root remain usable
  for detection/eval. Train frames are unchanged (full pose, `is_annotated=True`).
- **Calibration groups vs splits.** 11 calibration groups exist, but only those whose
  sequences are actually on disk in a split become sessions: train got
  ABF1/BB1/GPMF1/GSF1/MDF1/SB1/SMu4/ShSu1/SiBF1; AP1/MPM1 sequences live only in eval.
  SB1 appears in both (3 views in train, 2 in eval).
- **Resume/skip.** Added `session_is_complete` + `--overwrite`. Re-runs skip
  already-converted sessions (calib.npz present and frame count complete), so the
  combined `--split all` run skipped the 24 finished train sessions and only did eval.
- **Geometry fix during impl.** Initial `M_k = trans @ blkdiag(flip,1)` double-applied
  the GL→CV flip (flip is applied to the *points*, not folded into the matrix); fixed
  to `M_k = trans[slot]`, after which world agreement collapsed to < 1 mm. The plan's
  geometry note above reflects the original derivation; the script is the source of truth.

---

## Image storage decision: SYMLINK (run with `--symlink-images`)

We convert with `--symlink-images`, so `images/viewK/NNNNNN.jpg` are **symlinks**
to the original `ho3d/train/<seq>/rgb/*.jpg`, not copies. This saves tens of GB and
the long copy step. Reads are transparent — `cv2.imread`/PIL/PyTorch follow the
link, so WiLoR FT and any other consumer see identical pixels.

**This is safe for training only while these hold:**
- Training runs on a machine where the source NFS path
  `/lambda/nfs/hfm/qasim/hand_kp_dataset/ho3d/` is mounted identically (links store
  the **absolute** source path).
- The source `ho3d/train/` tree is not moved or deleted for the lifetime of training.
- The dataset is **not** tar'd/scp'd/rsync'd elsewhere without dereferencing
  (`rsync -L`, `tar --dereference`) — otherwise the links dangle and `imread`
  returns `None`.

**Re-run with copy (drop `--symlink-images`) if** you need a portable/standalone
dataset, or want the images on fast local scratch for I/O-bound FT. Symlink vs copy
does not change read *speed* on the same filesystem — both ultimately read from NFS;
only a physical copy to local disk makes reads faster.

## Context

We want HO-3D v3 hand annotations in the project's multi-view "session" format
(`session_xxx/calib.npz` + `frames/NNNNNN.npz` + `images/viewK/NNNNNN.jpg`) so it
sits alongside the existing converted datasets and is consumable the same way as
other hand-tracking data. HO-3D ships per-camera monocular sequences plus a
multi-camera calibration; we reconstruct true multi-view samples from it.

**This is a standalone conversion, not an extension of
[dataset.py](prometheus/src/metric_hand_tracking_v2/utils/dataset.py).** That file
is a set of runtime `torch.utils.data.Dataset` *loaders* (stereo rectification,
PCD/pose mixins, a `Hot3DDataset` VRS reader). A one-shot disk→disk converter has
no overlap with those classes and belongs in its own script under
`prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/`, matching the existing
[run_whim_to_wilor_cam.py](prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/run_whim_to_wilor_cam.py)
convention (argparse CLI, numpy-only, `__doc__` header explaining the geometry).

## Key facts established from the data

- **Groups vs sequences.** A multi-cam group `<G>` (e.g. `ABF1`, 11 groups:
  `ABF1, AP1, BB1, GPMF1, GSF1, MDF1, MPM1, SB1, SMu4, ShSu1, SiBF1`) has 5
  on-disk sequences `<G>0..<G>4`; the trailing digit `d` is the **camera id**.
  All 5 record the *same scene* (identical frame count, shared frame ids), each
  annotated in **its own camera's OpenGL frame**.
- **Single-cam sequences** (`MC*, ND2, SM*, SS*, SiS1, …`, `camIDList=['0']`,
  no calibration group) → 1-view sessions, view0 = world = identity extrinsics.
- **Calibration** at `calibration/<G>/calibration/`:
  `cam_<i>_intrinsics.txt` (`width,height,ppx,ppy,fx,fy,model,coeffs` — coeffs
  all 0, so `dist=zeros(5)`), `trans_<i>.txt` (4×4), `cam_orders.txt`
  (`order[i]` = camera id at slot `i`).
- **Verified geometry** (wrist reprojects to one world point from all 5 views):
  - `flip = diag(1,-1,-1)` maps OpenGL→OpenCV: `p_cv = flip @ p_gl`.
  - For sequence `<G>d`: slot `i = cam_orders.index(d)`; `T_i = trans_i` is
    **cam→world** in OpenCV: `p_world = T_i @ [p_cv;1]`.
  - So per-view `cam→world` = `T_i @ blkdiag(flip,1)` (fold the GL→CV flip in).
- **Per-frame meta** (`meta/NNNN.pkl`): `handPose(48,)` axis-angle (3 global +
  45), `handBeta(10,)`, `handTrans(3,)`, `handJoints3D(21,3)` (cam OpenGL frame),
  `camMat(3,3)`. Pose fields are `None` for unannotated frames. No 2D keypoints —
  project `handJoints3D` through `camMat` ourselves (after GL→CV flip).
- HO-3D is **right-hand only**; all hands `side="right"`, `hand_id=0`.

## Output format (per target spec)

```
ho3d_sessions/<split>/<session>/      # e.g. ho3d_sessions/train/ABF1, .../MC1
  calib.npz   { K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3),
                t_world_to_cam:(V,3), img_size:(V,2) }   # view0 = world frame
  frames/NNNNNN.npz   # one logical multi-view sample, keyed by HO-3D frame id
  images/view0/NNNNNN.jpg ... viewK/NNNNNN.jpg
```

`frames/NNNNNN.npz` (saved via `np.savez` with one object array `data`):
```python
{ "frame_id": int,
  "is_annotated": bool,           # False -> "hands": []  (lets us filter later)
  "hands": [ {
     "hand_id": 0, "side": "right",
     "mano": {"global_orient":(3,), "hand_pose":(45,), "betas":(10,)},  # world frame
     "views": { k: {"bbox":(4,), "joints_2d":(21,2), "joints_3d":(21,3)} } } ] }
```
- `joints_3d` per view = `handJoints3D` flipped to OpenCV, in **that view's cam frame**.
- `bbox` = tight xyxy around `joints_2d` (with small margin), clipped to img.
- `mano.global_orient` rebased into the **world (view0) frame**: compose view0's
  `R_world_to_cam` inverse with the per-view axis-angle global orient. Since each
  view stores the hand in its own cam frame, take view0's annotation as canonical:
  `R_global_world = R_view0_cam_to_world @ R(handPose[:3])`, re-encode to axis-angle.
  `hand_pose`/`betas` are view-independent → taken from view0.
- A view is included only if that sequence's frame is annotated; unannotated
  views are omitted from `views`. Frame-level `is_annotated = any view annotated`.

## Extrinsics rebasing (view0 = world)

For a group, build raw `cam→world` `M_k` for each view `k` (`M_k = T_{i_k} @
blkdiag(flip,1)`). Rebase so view0 is the world origin:
`M'_k = inv(M_0) @ M_k`. Then `world→cam` = `inv(M'_k)`; store
`R_world_to_cam = inv(M'_k)[:3,:3]`, `t_world_to_cam = inv(M'_k)[:3,3]`.
View0 → identity R, zero t. Single-cam sessions: V=1, identity.

## Script: `prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/ho3d_to_sessions.py`

Structure (numpy + opencv-for-imwrite only; mirror `run_whim_to_wilor_cam.py` style):

1. **CLI** (argparse): `--ho3d_root` (default the NFS path),
   `--out_root` (default sibling `ho3d_sessions/` next to source, or a path the
   user picks), `--split {train,all}` (eval has poses withheld → default `train`),
   `--copy-images/--symlink-images` (default copy), `--limit` seqs for smoke test.
2. **Discover sessions**: scan `train/`; group by stripping trailing digit when a
   matching `calibration/<G>/` exists (multi-cam), else treat sequence as its own
   1-view session. Produce `{session_name: [(view_k, seq_dir, cam_id), ...]}`.
3. **Build calib.npz** per session from intrinsics + trans files (helpers
   `load_intrinsics(path)`, `load_trans(i)`, `cam_orders`), applying the verified
   rebasing.
4. **Per frame** (union of frame ids across views; use the on-disk `rgb/*.jpg`
   intersection): load each view's pkl, flip joints to CV, project via `camMat`,
   compute bbox, assemble `hands`, write `frames/NNNNNN.npz`, and copy/symlink
   each view image to `images/viewK/NNNNNN.jpg`.
5. **Progress**: outer `tqdm` over sessions; inner `tqdm` over frames
   (`leave=False`). Print a one-line summary per session (views, #annotated/#total).
6. **Verification hook** (built into the script, `--verify` flag): for a sampled
   frame, reproject `joints_3d` (world→cam0 via stored extrinsics → K) and compare
   to stored `joints_2d`; assert mean px error < 1px. Also assert all views'
   `joints_3d` map to one world point (the invariant we already confirmed).

Reuse the existing axis-angle handling pattern; use
`scipy.spatial.transform.Rotation` (already a project dep, imported in dataset.py)
for the global-orient rebasing and matrix↔axis-angle conversions.

## Plan doc + post-run summary (in the converting dir)

- **Write this plan** to `prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/HO-3D_to_intermediate_format_conversion.md` (the
  design doc lives next to the script, not just in `~/.claude/plans`).
- **After the full conversion completes**, prepend a `## Summary (completed)`
  section at the **top** of that `plan.md` recording: date, what was run, output
  location, session/frame/view counts produced, verify results (px/mm errors),
  and any key changes or deviations from the plan discovered during
  implementation (e.g. pkl dtype quirks, missing-frame handling, side cases).

## Critical files

- **Create**: `prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/ho3d_to_sessions.py`
- **Create**: `prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/HO-3D_to_intermediate_format_conversion.md` (this design doc;
  gets a `## Summary (completed)` header prepended after the run)
- **Reference (style only)**:
  [run_whim_to_wilor_cam.py](prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/run_whim_to_wilor_cam.py)
- **Do not modify**:
  [dataset.py](prometheus/src/metric_hand_tracking_v2/utils/dataset.py)
  (separate concern; a runtime loader for the new format, if wanted, is a future
  follow-up, not part of this conversion).
- **Source data**: `/lambda/nfs/hfm/qasim/hand_kp_dataset/ho3d/`

## Verification (end-to-end)

1. Smoke run on one multi-cam group + one single-cam seq:
   `python ho3d_to_sessions.py --split train --limit 2 --verify --out_root /tmp/ho3d_sessions`
2. Check structure: `calib.npz` shapes (`V` views), a `frames/000000.npz` loads
   and contains a right hand with 5 views (ABF1) / 1 view (MC1).
3. `--verify` asserts: (a) world-point agreement across views < 1mm,
   (b) reprojection of stored `joints_3d` vs `joints_2d` < 1px.
4. Visual spot check: overlay `joints_2d` on `images/view0/000000.jpg` for one
   frame and eyeball that keypoints land on the hand.
5. Full run after smoke passes; confirm session/frame counts roughly match
   train.txt's 83,325 frame-ids (annotated subset) plus unannotated frames flagged.

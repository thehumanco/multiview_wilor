# Convert InterHand2.6M -> multi-view session format

Script: [`interhand_to_sessions.py`](interhand_to_sessions.py)

## Summary (completed)

Full conversion was run once on 2026-06-09, but that run preserved InterHand's
native keypoint order. It should be rerun with the current converter and
`--overwrite` so `joints_2d`/`joints_3d` match HO-3D's intermediate format.

Smoke verification was completed on 2026-06-09 after qasim-owned annotation JSONs
were made world-readable.

Smoke command:

```bash
conda activate prometheus
python \
  src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/interhand_to_sessions.py \
  --split val --limit 1 --verify --symlink-images \
  --out_root /tmp/interhand_sessions --overwrite
```

Smoke output:

- Session: `val/Capture0__ROM01_No_Interaction_2_Hand`
- Views: 139
- Frames: 124
- Annotated frames: 103
- Hand instances: 149
- `--verify`: passed reprojection `<1 px` and multi-view world agreement `<1 mm`
- Output images are absolute symlinks to the source JPGs
- Visual spot check written to `/tmp/interhand_sessions/val_overlay_view0.jpg`;
  projected joints land on the hands

After the full run, record the output path, per-split session/frame/view counts,
aggregate verification errors, and any full-run deviations.

## Purpose

Convert InterHand2.6M 5fps batch1 hand annotations into the project's multi-view
"session" format:

```
<out_root>/<split>/Capture<X>__<seq_name>/
  calib.npz
  frames/NNNNNN.npz
  images/view0/NNNNNN.jpg ... viewK/NNNNNN.jpg
```

This is a standalone disk-to-disk converter under
`prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/`. It does not modify or extend
`src/metric_hand_tracking_v2/utils/dataset.py`.

## Key Decisions

- Session unit is `(Capture, sequence)`, emitted as `Capture<X>__<seq_name>`.
- Images default to absolute symlinks because copying all views would be enormous.
  Use `--copy-images` only when a portable dataset is needed.
- All on-disk frames are emitted. Frames with NeuralAnnot MANO for at least one
  visible hand get `is_annotated=True`; missing MANO is zero-filled and marked
  `is_annotated=False`.
- InterHand's native per-hand 21-joint order is remapped to the HO-3D/project
  intermediate keypoint order: wrist; index/middle/pinky/ring/thumb MANO joints;
  then thumb/index/middle/ring/pinky fingertips. MANO `hand_pose` remains in
  standard MANO pose-vector order.
- Field-order audit against HO-3D:
  - `calib.npz`: no change. Same keys/shapes; view0 is the world frame.
  - `frame_id`, `hand_id`, `side`, `views`: no change. Same structure.
  - `mano.global_orient`: no change. Rebased to view0 world frame, like HO-3D.
  - `mano.hand_pose`: no remap. It stays in native MANO 15-joint axis-angle
    order, matching HO-3D `handPose[3:]` and OakInk `pose_coeffs[1:]`.
  - `mano.betas`: no change. First 10 shape coefficients.
  - `bbox`: no order issue; xyxy, clipped, 15% margin.
  - `joints_2d` and `joints_3d`: changed from InterHand native order to the
    HO-3D session order below.

## Source Schema Assumptions

The converter targets the canonical InterHand/NeuralAnnot JSON layout:

- `InterHand2.6M_<split>_data.json`: COCO-style `images` and `annotations`.
  Images provide `capture`, `seq_name`, `camera`, `frame_idx`, dimensions, and
  `file_name`.
- `InterHand2.6M_<split>_camera.json`: keyed by capture, then `campos`, `camrot`,
  `focal`, and `princpt`, each keyed by camera id.
- `InterHand2.6M_<split>_joint_3d.json`: keyed by capture then frame id, with
  `world_coord` `(42,3)` in millimeters and `joint_valid` `(42,)`.
- `InterHand2.6M_<split>_MANO_NeuralAnnot.json`: keyed by capture then frame id,
  with optional `right` and `left` MANO dicts containing `pose`, `shape`, and
  `trans`.

The annotation directory must be readable by the user running the converter. In
the current checkout these JSONs were observed as qasim-owned mode `600`; they were
changed to mode `604` with:

```bash
chmod -R o+r /lambda/nfs/hfm/qasim/hand_kp_dataset/interhands2.6m/annotations
```

Schema dump confirmed:

- `data.json` has `images` and `annotations`; val has 380,125 of each.
- Sample image dimensions are `width=334`, `height=512`.
- `joint_valid` is stored as a `(42,1)` nested list.
- `camera.json` fields are `campos`, `camrot`, `focal`, `princpt`.
- `joint_3d.json` has `world_coord`, `joint_valid`, `hand_type`,
  `hand_type_valid`, and `seq`.
- MANO entries contain optional `right`/`left` dicts with `pose` `(48,)`,
  `shape` `(10,)`, and `trans` `(3,)`.

## Geometry

- `camrot` is treated as world-to-camera rotation.
- `campos` is treated as camera position in world millimeters.
- Raw world-to-camera translation is `t = -camrot @ campos * 1e-3`.
- Camera ids are sorted; the smallest camera id becomes `view0`.
- Extrinsics are rebased so `view0` is the session world:
  `E'_k = E_k @ inv(E_0)`.
- World joints are converted from millimeters to meters and mapped into the new
  world frame with `E_0`.
- Per-view `joints_3d` is stored in that view's camera frame; `joints_2d` is
  projected from `K`; `bbox` is tight xyxy with a 15% margin.
- `joints_2d` and `joints_3d` use the same 21-keypoint order as HO-3D:
  `[wrist, thumb1, thumb2, thumb3, thumb4, index1, index2, index3, index4,
  middle1, middle2, middle3, middle4, ring1, ring2, ring3, ring4,
  pinky1, pinky2, pinky3, pinky4]`.
- `mano.hand_pose` uses the native MANO 15-joint pose-vector order, not the
  21-keypoint array order:
  `[index1, index2, index3, middle1, middle2, middle3, pinky1, pinky2, pinky3,
  ring1, ring2, ring3, thumb1, thumb2, thumb3]`.
- MANO `pose[:3]` global orientation is rebased into the `view0` world frame.

## Output Frame Payload

Each `frames/NNNNNN.npz` stores `data=np.array(data_dict, dtype=object)`:

```python
{
  "frame_id": int,
  "is_annotated": bool,
  "hands": [
    {
      "hand_id": 0 | 1,
      "side": "right" | "left",
      "mano": {
        "global_orient": (3,),
        "hand_pose": (45,),
        "betas": (10,),
      },
      "views": {
        k: {
          "bbox": (4,),
          "joints_2d": (21, 2),
          "joints_3d": (21, 3),
        }
      },
    }
  ],
}
```

Per-hand views are included only when the side's joints are valid, all depths are
positive, and at least 50% of joints project inside the image.

## Verification

Run a smoke conversion after annotation access is fixed:

```bash
python prometheus/src/metric_hand_tracking_v2/wilor_v2/datasets/data_processing/converting/interhand_to_sessions.py \
  --split val \
  --limit 1 \
  --verify \
  --symlink-images \
  --out_root /tmp/interhand_sessions
```

The `--verify` path checks:

- Stored per-view `joints_3d` reproject through `K` to stored `joints_2d` within
  1 pixel.
- Multi-view hands map back to one rebased world point within 1 mm.

Also inspect one frame manually and overlay `joints_2d` on an image for a visual
spot check before launching the full conversion.

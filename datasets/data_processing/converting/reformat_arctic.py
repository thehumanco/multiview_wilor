#!/usr/bin/env python3
"""Reformat Arctic dataset into multiview pinhole format.

Output layout:
  out_dir/
    train/
      s01_box_grab_01/
        calib.npz               # K, dist, R_world_to_cam, t_world_to_cam, img_size
        frames/000000.npz       # {'frame_id', 'is_annotated', 'hands': [...]}
        images/view0/000000.jpg
        images/view1/000000.jpg
        ...
    evaluation/
      s04_box_grab_01/
        ...

calib.npz arrays are stacked over the 8 static allocentric views + 1 egocentric view (index 8):
  K:              (9, 3, 3)
  dist:           (9, 5)            all zeros (pinhole projection)
  R_world_to_cam: (9, 3, 3)        view 8 (ego) is a frame-0 placeholder; see dynamic_views
  t_world_to_cam: (9, 3)
  img_size:       (9, 2)            [width, height]
  dynamic_views:  (1,) = [8]        ego view: per-frame world->cam lives in each frame npz

The egocentric camera (camera folder "0" in the raw data) is head-mounted and moves every
frame; its per-frame world->cam [R|t] is written into the frame npz under
views[8]["extrinsics"], and the dataloader reads it from there instead of calib.npz.

Each frames/*.npz has a single 'data' key holding a Python dict:
  {
    'frame_id': int,
    'is_annotated': bool,
    'hands': [
      {
        'hand_id': int,           0 = right, 1 = left
        'side': 'right'|'left',
        'mano': {
          'global_orient': (3,)   axis-angle, world frame
          'hand_pose':     (45,)  full pose params (use_pca=False)
          'betas':         (10,)  shape params
          'translation':   (3,)   world frame, metres
        },
        'views': {
          <int view_idx>: {
            'bbox':      (4,)    [x_min, y_min, x_max, y_max]
            'joints_2d': (21,2)  image space, standard MANO joint order
            'joints_3d': (21,3)  camera frame, standard MANO joint order
            'extrinsics':(3,4)   ONLY for the ego view (8): per-frame world->cam [R|t]
          },
          ...
        }
      },
      ...
    ]
  }

Joint order (standard MANO — smplx kinematic chain + fingertip vertices):
  0: wrist
  1-3:   index  mcp/pip/dip
  4-6:   middle mcp/pip/dip
  7-9:   pinky  mcp/pip/dip
  10-12: ring   mcp/pip/dip
  13-15: thumb  cmc/mcp/ip
  16: thumb_tip  17: index_tip  18: middle_tip
  19: ring_tip   20: pinky_tip

Eval split: subjects s03 and s04 (Arctic p1 test subjects).
  s03 raw sequences are not present in this download; only s04 data is written.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch
from smplx import MANO

DATA_DIR = "/lambda/nfs/hfm/qasim/hand_kp_dataset/arctic/unpack/arctic_data/data"
MANO_DIR = "/lambda/nfs/hfm/qasim/hand_kp_dataset/arctic/unpack/body_models/mano"
OUT_DIR = "/lambda/nfs/hfm/qasim/hand_kp_dataset/arctic/sessions"

# Arctic p1 test subjects held out for evaluation.
EVAL_SUBJECTS = {"s03", "s04"}

N_STATIC_CAMS = 8  # static view folders 1-8 in images/<sid>/<seq>/

MANO_BATCH = 256

# Fingertip vertex IDs in MANO mesh (778 verts), in joint-order:
# thumb(16), index(17), middle(18), ring(19), pinky(20).
# Matches the smplx vertex_joint_selector ordering for MANO.
FINGERTIP_VERTEX_IDS = [744, 320, 443, 554, 671]


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def session_is_complete(session_dir: str, n_frames: int) -> bool:
    if not os.path.isfile(os.path.join(session_dir, "calib.npz")):
        return False
    return len(glob.glob(os.path.join(session_dir, "frames", "*.npz"))) >= n_frames


# ---------------------------------------------------------------------------
# MANO helpers
# ---------------------------------------------------------------------------

def _patch_chumpy():
    """Patch inspect so older chumpy/smplx pkl files load on Python 3.11+."""
    import inspect as _inspect
    if not hasattr(_inspect, "getargspec"):
        _inspect.getargspec = _inspect.getfullargspec


def build_mano(is_right: bool, device: torch.device) -> MANO:
    return _build_mano_from_dir(is_right, device, MANO_DIR)


def _build_mano_from_dir(is_right: bool, device: torch.device, mano_dir: str) -> MANO:
    _patch_chumpy()
    model = MANO(
        mano_dir,
        create_transl=False,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=is_right,
    )
    return model.to(device).eval()


def run_mano(model: MANO, rot, pose, shape, trans, device: torch.device) -> np.ndarray:
    """Forward MANO for all F frames. Returns (F, 21, 3) joints in world space.

    The 16 kinematic-chain joints from smplx are augmented with 5 fingertip
    vertices (thumb/index/middle/ring/pinky at mesh indices 744/320/443/554/671),
    matching the order that smplx's vertex_joint_selector would produce.
    """
    F = rot.shape[0]
    all_joints = []
    with torch.no_grad():
        for i in range(0, F, MANO_BATCH):
            r = torch.tensor(rot[i:i + MANO_BATCH], dtype=torch.float32, device=device)
            p = torch.tensor(pose[i:i + MANO_BATCH], dtype=torch.float32, device=device)
            b = torch.tensor(shape[i:i + MANO_BATCH], dtype=torch.float32, device=device)
            out = model(global_orient=r, hand_pose=p, betas=b)
            j = out.joints.cpu().numpy()    # (B, 16, 3) kinematic joints
            v = out.vertices.cpu().numpy()  # (B, 778, 3) mesh vertices
            tips = v[:, FINGERTIP_VERTEX_IDS, :]          # (B, 5, 3)
            j = np.concatenate([j, tips], axis=1)         # (B, 21, 3)
            j = j + trans[i:i + MANO_BATCH, None, :]      # add world translation
            all_joints.append(j.astype(np.float32))
    return np.concatenate(all_joints, axis=0)  # (F, 21, 3)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def project(K, R, t, joints_world: np.ndarray):
    """
    K: (3, 3)  intrinsics
    R: (3, 3), t: (3,)  world-to-camera rotation and translation
    joints_world: (..., 3)

    Returns:
      joints_cam: same shape as input, in camera frame (float32)
      joints_2d:  same shape but last dim 2, pixel coords (float32)
    """
    orig_shape = joints_world.shape
    pts = joints_world.reshape(-1, 3)            # (N, 3)
    pts_cam = (R @ pts.T).T + t                  # (N, 3)

    Z = pts_cam[:, 2:3].clip(1e-6, None)
    xy = pts_cam[:, :2] / Z
    u = xy[:, 0] * K[0, 0] + K[0, 2]
    v = xy[:, 1] * K[1, 1] + K[1, 2]

    joints_cam = pts_cam.reshape(orig_shape).astype(np.float32)
    joints_2d = np.stack([u, v], axis=-1).reshape(*orig_shape[:-1], 2).astype(np.float32)
    return joints_cam, joints_2d


def bbox_from_joints2d(joints_2d: np.ndarray, img_w: int, img_h: int, pad: int = 10):
    """Return [x_min, y_min, x_max, y_max] (float32) or None.

    A view is considered valid only when the hand is substantially visible:
    the wrist (joint 0) must be inside the image, and at least half of the
    21 joints must lie within image bounds.  This filters out frames where
    a hand's projection merely clips the edge of a camera that isn't
    actually looking at that hand.
    """
    in_img = (
        (joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < img_w) &
        (joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < img_h)
    )
    if not in_img[0] or in_img.sum() < 11:
        return None
    valid = joints_2d[in_img]
    x0 = max(0.0, float(valid[:, 0].min()) - pad)
    y0 = max(0.0, float(valid[:, 1].min()) - pad)
    x1 = min(float(img_w), float(valid[:, 0].max()) + pad)
    y1 = min(float(img_h), float(valid[:, 1].max()) + pad)
    if x0 >= x1 or y0 >= y1:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_sequence(
    sid: str,
    seq_name: str,
    misc: dict,
    mano_r: MANO,
    mano_l: MANO,
    device: torch.device,
    session_dir: str,
    skip_images: bool,
    data_dir: str = DATA_DIR,
) -> int:
    mano_path = os.path.join(data_dir, "raw_seqs", sid, f"{seq_name}.mano.npy")
    mano_data = np.load(mano_path, allow_pickle=True).item()

    # Unpack per-side MANO params; broadcast scalar shape to (F, 10).
    def unpack_side(d):
        F = d["rot"].shape[0]
        shape = d["shape"]
        if shape.ndim == 1:
            shape = np.tile(shape[None], (F, 1))
        return (
            d["rot"].astype(np.float32),    # (F, 3)
            d["pose"].astype(np.float32),   # (F, 45)
            shape.astype(np.float32),       # (F, 10)
            d["trans"].astype(np.float32),  # (F, 3)
        )

    rot_r, pose_r, shape_r, trans_r = unpack_side(mano_data["right"])
    rot_l, pose_l, shape_l, trans_l = unpack_side(mano_data["left"])
    F = rot_r.shape[0]

    ioi_offset = int(misc[sid]["ioi_offset"])
    world2cam = np.array(misc[sid]["world2cam"], dtype=np.float64)  # (8, 4, 4)
    intris = np.array(misc[sid]["intris_mat"], dtype=np.float64)    # (8, 3, 3)
    img_sizes = np.array(misc[sid]["image_size"], dtype=np.int64)   # (9, 2): [ego, cam1..cam8]

    # ---- ego camera (per-frame extrinsics) ----------------------------------
    # The head-mounted egocentric camera moves every frame, so its world->cam is stored per
    # frame in the frame npz ("extrinsics"); calib.npz holds only the static ego intrinsics +
    # a frame-0 placeholder R/t. Ego becomes the LAST view (index N_STATIC_CAMS).
    ego_path = os.path.join(data_dir, "raw_seqs", sid, f"{seq_name}.egocam.dist.npy")
    egocam = np.load(ego_path, allow_pickle=True).item()
    R_ego = np.asarray(egocam["R_k_cam_np"], dtype=np.float64)             # (F, 3, 3) world->cam
    t_ego = np.asarray(egocam["T_k_cam_np"], dtype=np.float64)[:, :, 0]    # (F, 3)
    K_ego = np.asarray(egocam["intrinsics"], dtype=np.float64)            # (3, 3) static
    EGO_VIEW = N_STATIC_CAMS
    ego_w, ego_h = int(img_sizes[0, 0]), int(img_sizes[0, 1])

    # ---- calib.npz ----------------------------------------------------------
    Ks, dists, Rs, ts, sizes = [], [], [], [], []
    for v in range(N_STATIC_CAMS):
        W = world2cam[v]  # (4, 4) world-to-cam
        Ks.append(intris[v])
        dists.append(np.zeros(5, dtype=np.float64))
        Rs.append(W[:3, :3])
        ts.append(W[:3, 3])
        # img_sizes[0] = ego; img_sizes[v+1] = static cam v+1
        sizes.append(img_sizes[v + 1])
    # ego view: static K, frame-0 placeholder extrinsics, ego image size.
    Ks.append(K_ego)
    dists.append(np.zeros(5, dtype=np.float64))  # dist unused by the loader; keep width 5
    Rs.append(R_ego[0])
    ts.append(t_ego[0])
    sizes.append(img_sizes[0])

    os.makedirs(session_dir, exist_ok=True)
    np.savez(
        os.path.join(session_dir, "calib"),
        K=np.stack(Ks),
        dist=np.stack(dists),
        R_world_to_cam=np.stack(Rs),
        t_world_to_cam=np.stack(ts),
        img_size=np.stack(sizes),
        dynamic_views=np.array([EGO_VIEW], dtype=np.int64),
    )

    # ---- MANO forward pass for all frames -----------------------------------
    j3d_r_world = run_mano(mano_r, rot_r, pose_r, shape_r, trans_r, device)  # (F,21,3)
    j3d_l_world = run_mano(mano_l, rot_l, pose_l, shape_l, trans_l, device)  # (F,21,3)

    # ---- Project to all static cameras (batched over F) ---------------------
    cam_data_r = []  # list of (j3d_cam (F,21,3), j2d (F,21,2)) per view
    cam_data_l = []
    for v in range(N_STATIC_CAMS):
        K = intris[v]
        R = world2cam[v, :3, :3]
        t = world2cam[v, :3, 3]
        j3d_r_cam, j2d_r = project(K, R, t, j3d_r_world)
        j3d_l_cam, j2d_l = project(K, R, t, j3d_l_world)
        cam_data_r.append((j3d_r_cam, j2d_r))
        cam_data_l.append((j3d_l_cam, j2d_l))

    # ---- Output directories -------------------------------------------------
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    if not skip_images:
        for v in range(N_STATIC_CAMS):
            os.makedirs(os.path.join(session_dir, "images", f"view{v}"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "images", f"view{EGO_VIEW}"), exist_ok=True)

    # ---- Per-frame files ----------------------------------------------------
    img_base = os.path.join(data_dir, "images", sid, seq_name)
    for frame_idx in range(F):
        views_r, views_l = {}, {}
        img_num = frame_idx + ioi_offset
        for v in range(N_STATIC_CAMS):
            img_w, img_h = int(img_sizes[v + 1, 0]), int(img_sizes[v + 1, 1])
            src = os.path.join(img_base, str(v + 1), f"{img_num:05d}.jpg")
            if not os.path.exists(src):
                # Raw image extraction is short a few frames for this view (e.g. arctic
                # s06_notebook_use_01 is missing the last 7 frames on every camera); skip
                # rather than write an annotation with no backing image.
                continue

            j3d_r_cam_f = cam_data_r[v][0][frame_idx]  # (21, 3)
            j2d_r_f = cam_data_r[v][1][frame_idx]       # (21, 2)
            j3d_l_cam_f = cam_data_l[v][0][frame_idx]
            j2d_l_f = cam_data_l[v][1][frame_idx]

            bbox_r = bbox_from_joints2d(j2d_r_f, img_w, img_h)
            if bbox_r is not None:
                views_r[v] = {
                    "bbox": bbox_r,
                    "joints_2d": j2d_r_f,
                    "joints_3d": j3d_r_cam_f,
                }

            bbox_l = bbox_from_joints2d(j2d_l_f, img_w, img_h)
            if bbox_l is not None:
                views_l[v] = {
                    "bbox": bbox_l,
                    "joints_2d": j2d_l_f,
                    "joints_3d": j3d_l_cam_f,
                }

        # ego view: project with this frame's world->cam, store extrinsics in the view entry.
        src_e = os.path.join(img_base, "0", f"{img_num:05d}.jpg")
        if os.path.exists(src_e):
            R_e, t_e = R_ego[frame_idx], t_ego[frame_idx]
            ext_e = np.concatenate([R_e, t_e[:, None]], axis=1).astype(np.float32)  # (3,4)
            j3d_r_e, j2d_r_e = project(K_ego, R_e, t_e, j3d_r_world[frame_idx])
            j3d_l_e, j2d_l_e = project(K_ego, R_e, t_e, j3d_l_world[frame_idx])

            bbox_r_e = bbox_from_joints2d(j2d_r_e, ego_w, ego_h)
            if bbox_r_e is not None:
                views_r[EGO_VIEW] = {
                    "bbox": bbox_r_e,
                    "joints_2d": j2d_r_e,
                    "joints_3d": j3d_r_e,
                    "extrinsics": ext_e,
                }
            bbox_l_e = bbox_from_joints2d(j2d_l_e, ego_w, ego_h)
            if bbox_l_e is not None:
                views_l[EGO_VIEW] = {
                    "bbox": bbox_l_e,
                    "joints_2d": j2d_l_e,
                    "joints_3d": j3d_l_e,
                    "extrinsics": ext_e,
                }

        hands = []
        if views_r:
            hands.append({
                "hand_id": 0,
                "side": "right",
                "mano": {
                    "global_orient": rot_r[frame_idx],
                    "hand_pose": pose_r[frame_idx],
                    "betas": shape_r[frame_idx],
                    "translation": trans_r[frame_idx],
                },
                "views": views_r,
            })
        if views_l:
            hands.append({
                "hand_id": 1,
                "side": "left",
                "mano": {
                    "global_orient": rot_l[frame_idx],
                    "hand_pose": pose_l[frame_idx],
                    "betas": shape_l[frame_idx],
                    "translation": trans_l[frame_idx],
                },
                "views": views_l,
            })

        frame_data = {
            "frame_id": frame_idx,
            "is_annotated": len(hands) > 0,
            "hands": hands,
        }
        np.savez(
            os.path.join(frames_dir, f"{frame_idx:06d}"),
            data=np.array(frame_data, dtype=object),
        )

        if not skip_images:
            for v in range(N_STATIC_CAMS):
                src = os.path.join(img_base, str(v + 1), f"{img_num:05d}.jpg")
                dst = os.path.join(session_dir, "images", f"view{v}", f"{frame_idx:06d}.jpg")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            # ego images live under camera folder "0".
            if os.path.exists(src_e):
                dst_e = os.path.join(session_dir, "images", f"view{EGO_VIEW}", f"{frame_idx:06d}.jpg")
                shutil.copy2(src_e, dst_e)

    return F


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reformat Arctic into multiview pinhole format."
    )
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--mano_dir", default=MANO_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--debug", action="store_true",
                        help="Process only the first 2 sequences.")
    parser.add_argument("--skip_images", action="store_true",
                        help="Skip copying colour JPEGs (faster for testing).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert already-complete sessions (default: skip them).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = args.data_dir
    mano_dir = args.mano_dir

    with open(os.path.join(data_dir, "meta", "misc.json")) as f:
        misc = json.load(f)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    mano_r = _build_mano_from_dir(is_right=True, device=device, mano_dir=mano_dir)
    mano_l = _build_mano_from_dir(is_right=False, device=device, mano_dir=mano_dir)

    # Collect all sequences: raw_seqs/<sid>/<seq>.mano.npy
    raw_seqs_dir = os.path.join(data_dir, "raw_seqs")
    sequences = []  # list of (sid, seq_name)
    for sid in sorted(os.listdir(raw_seqs_dir)):
        sid_dir = os.path.join(raw_seqs_dir, sid)
        if not os.path.isdir(sid_dir):
            continue
        for fname in sorted(os.listdir(sid_dir)):
            if fname.endswith(".mano.npy"):
                seq_name = fname[: -len(".mano.npy")]
                sequences.append((sid, seq_name))

    if args.debug:
        sequences = sequences[:2]

    n_total = len(sequences)
    n_eval = sum(1 for sid, _ in sequences if sid in EVAL_SUBJECTS)
    n_train = n_total - n_eval
    print(f"Total sequences: {n_total}  |  train: {n_train}  |  eval: {n_eval}")

    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "evaluation"), exist_ok=True)

    n_skipped = 0
    for idx, (sid, seq_name) in enumerate(sequences):
        split = "evaluation" if sid in EVAL_SUBJECTS else "train"
        session_name = f"{sid}_{seq_name}"
        session_dir = os.path.join(args.out_dir, split, session_name)
        print(
            f"[{idx + 1:04d}/{n_total}] [{split[:5]}] {sid}/{seq_name} → {session_name}",
            flush=True,
        )

        if not args.overwrite:
            mano_path = os.path.join(data_dir, "raw_seqs", sid, f"{seq_name}.mano.npy")
            mano_meta = np.load(mano_path, allow_pickle=True).item()
            F = int(mano_meta["right"]["rot"].shape[0])
            if session_is_complete(session_dir, F):
                print(f"          skipped (already complete: {F} frames)", flush=True)
                n_skipped += 1
                continue

        n_frames = process_sequence(
            sid, seq_name, misc, mano_r, mano_l, device, session_dir,
            args.skip_images, data_dir=data_dir,
        )
        print(f"          {n_frames} frames written", flush=True)

    print(f"Done. ({n_skipped}/{n_total} sessions skipped as already complete)")


if __name__ == "__main__":
    main()

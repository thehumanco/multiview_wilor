#!/usr/bin/env python3
"""Reformat DexYCB dataset into multiview pinhole format.

Output layout:
  out_dir/
    train/
      s01_20200709_141754/
        calib.npz               # K, dist, R_world_to_cam, t_world_to_cam, img_size
        frames/000000.npz       # {'frame_id', 'is_annotated', 'hands': [...]}
        images/view0/000000.jpg
        images/view1/000000.jpg
        ...
    evaluation/
      s02_20200813_100608/
        ...

calib.npz arrays are stacked over views:
  K:             (n_views, 3, 3)
  dist:          (n_views, 5)
  R_world_to_cam:(n_views, 3, 3)   view 0 = master = world frame (identity)
  t_world_to_cam:(n_views, 3)
  img_size:      (n_views, 2)

Each frames/*.npz has a single 'data' key holding a Python dict:
  {
    'frame_id': int,
    'is_annotated': bool,
    'hands': [
      {
        'hand_id': int,
        'side': 'right'|'left',
        'mano': {
          'global_orient': (3,)  axis-angle, world frame
          'hand_pose':     (45,) PCA coefficients
          'betas':         (10,) shape params
          'translation':   (3,)  world frame (metres)
        },
        'views': {
          <int view_idx>: {
            'bbox':      (4,)    [x_min, y_min, x_max, y_max]
            'joints_2d': (21,2)  image space,  standard MANO order (see JOINT_REORDER)
            'joints_3d': (21,3)  camera frame, standard MANO order
          },
          ...
        }
      }
    ]
  }

Eval split: min(5% of sessions, 8 sessions), one per subject, subjects selected
to spread across the dataset.
"""

import argparse
import os
import shutil

import numpy as np
import yaml

DATA_DIR = "/lambda/nfs/hfm/qasim/hand_kp_dataset/dexycb"
OUT_DIR = "/lambda/nfs/hfm/qasim/hand_kp_dataset/dexycb/processed/multiview_pinhole"

SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

IMG_W, IMG_H = 640, 480

# Permutation that converts DexYCB joint order to the standard MANO order:
#   DexYCB: wrist, thumb(mcp pip dip tip), index(mcp pip dip tip),
#           middle(mcp pip dip tip), ring(mcp pip dip tip), little(mcp pip dip tip)
#   Target: wrist,
#           index(mcp pip dip), middle(mcp pip dip), pinky(mcp pip dip), ring(mcp pip dip),
#           thumb(cmc mcp ip tip), index_tip, middle_tip, ring_tip, pinky_tip
JOINT_REORDER = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_intrinsics(calib_dir, serial):
    path = os.path.join(calib_dir, "intrinsics", f"{serial}_{IMG_W}x{IMG_H}.yml")
    with open(path) as f:
        intr = yaml.load(f, Loader=yaml.FullLoader)
    c = intr["color"]
    K = np.array(
        [[c["fx"], 0.0, c["ppx"]], [0.0, c["fy"], c["ppy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def load_extrinsics(calib_dir, extr_key):
    """Returns {serial: 3×4 cam-to-world matrix} and master serial."""
    path = os.path.join(calib_dir, f"extrinsics_{extr_key}", "extrinsics.yml")
    with open(path) as f:
        extr = yaml.load(f, Loader=yaml.FullLoader)
    mats = {
        s: np.array(v, dtype=np.float64).reshape(3, 4)
        for s, v in extr["extrinsics"].items()
        if s != "apriltag"
    }
    return mats, extr["master"]


def cam2world_to_world2cam(R_c2w, t_c2w):
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    return R_w2c, t_w2c


def bbox_from_joints2d(joints_2d, pad=10):
    """Returns [x_min, y_min, x_max, y_max] or None if no in-image joints."""
    in_image = (
        (joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < IMG_W) &
        (joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < IMG_H)
    )
    valid = joints_2d[in_image]
    if len(valid) == 0:
        return None
    x_min = max(0.0, float(valid[:, 0].min()) - pad)
    y_min = max(0.0, float(valid[:, 1].min()) - pad)
    x_max = min(float(IMG_W), float(valid[:, 0].max()) + pad)
    y_max = min(float(IMG_H), float(valid[:, 1].max()) + pad)
    if x_min >= x_max or y_min >= y_max:
        return None
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def session_name_from_path(seq_path):
    """s01_20200709_141754 from .../20200709-subject-01/20200709_141754."""
    parts = seq_path.rstrip("/").split(os.sep)
    timestamp = parts[-1]          # e.g. 20200709_141754
    subject = parts[-2]            # e.g. 20200709-subject-01
    num = subject.split("-subject-")[-1]   # e.g. 01
    return f"s{num}_{timestamp}"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def build_calib(serials, calib_dir, extr_mats, master_serial):
    """Return stacked calib arrays with view-0 = master (world frame)."""
    ordered = [master_serial] + [s for s in serials if s != master_serial]
    Ks, dists, Rs, ts, sizes = [], [], [], [], []
    for serial in ordered:
        K, dist = load_intrinsics(calib_dir, serial)
        R_c2w = extr_mats[serial][:, :3]
        t_c2w = extr_mats[serial][:, 3]
        R_w2c, t_w2c = cam2world_to_world2cam(R_c2w, t_c2w)
        Ks.append(K)
        dists.append(dist)
        Rs.append(R_w2c)
        ts.append(t_w2c)
        sizes.append(np.array([IMG_W, IMG_H], dtype=np.int64))
    return ordered, {
        "K":             np.stack(Ks),
        "dist":          np.stack(dists),
        "R_world_to_cam": np.stack(Rs),
        "t_world_to_cam": np.stack(ts),
        "img_size":      np.stack(sizes),
    }


def process_sequence(seq_path, session_dir, calib_dir, skip_images):
    meta_file = os.path.join(seq_path, "meta.yml")
    with open(meta_file) as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    serials = meta["serials"]
    num_frames = meta["num_frames"]
    extr_key = meta["extrinsics"]
    mano_sides = meta["mano_sides"]
    mano_calib_names = meta["mano_calib"]

    extr_mats, master_serial = load_extrinsics(calib_dir, extr_key)
    ordered_serials, calib_arrays = build_calib(serials, calib_dir, extr_mats, master_serial)
    serial_to_view = {s: i for i, s in enumerate(ordered_serials)}
    n_views = len(ordered_serials)

    # ---- calib.npz --------------------------------------------------------
    os.makedirs(session_dir, exist_ok=True)
    np.savez(os.path.join(session_dir, "calib"), **calib_arrays)

    # ---- MANO betas -------------------------------------------------------
    mano_betas = []
    for calib_name in mano_calib_names:
        path = os.path.join(calib_dir, f"mano_{calib_name}", "mano.yml")
        with open(path) as f:
            mc = yaml.load(f, Loader=yaml.FullLoader)
        mano_betas.append(np.array(mc["betas"], dtype=np.float32))

    # ---- World-frame MANO poses (pose.npz = world frame) ------------------
    pose_m_world = np.load(os.path.join(seq_path, "pose.npz"))["pose_m"]  # (F, H, 51)

    # ---- Create output dirs -----------------------------------------------
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    if not skip_images:
        for i in range(n_views):
            os.makedirs(os.path.join(session_dir, "images", f"view{i}"), exist_ok=True)

    # ---- Per-frame processing ---------------------------------------------
    n_hands = len(mano_sides)
    frames_written = 0

    for frame_idx in range(num_frames):
        # Gather per-view observations.
        views_per_hand = [{} for _ in range(n_hands)]
        for serial in serials:
            view_idx = serial_to_view[serial]
            label_file = os.path.join(seq_path, serial, f"labels_{frame_idx:06d}.npz")
            if not os.path.exists(label_file):
                continue
            label = np.load(label_file)
            for hand_idx in range(n_hands):
                joint_3d = label["joint_3d"][hand_idx]  # (21, 3), DexYCB order
                joint_2d = label["joint_2d"][hand_idx]  # (21, 2), DexYCB order
                if np.all(joint_3d == -1):
                    continue
                bbox = bbox_from_joints2d(joint_2d)
                if bbox is None:
                    continue
                # Reorder joints to standard MANO order.
                views_per_hand[hand_idx][view_idx] = {
                    "bbox": bbox,
                    "joints_2d": joint_2d[JOINT_REORDER].astype(np.float32),
                    "joints_3d": joint_3d[JOINT_REORDER].astype(np.float32),
                }

        # Build hands list.
        hands = []
        for hand_idx, (side, betas) in enumerate(zip(mano_sides, mano_betas)):
            if not views_per_hand[hand_idx]:
                continue
            pm = pose_m_world[frame_idx, hand_idx]  # (51,)
            if np.all(pm == 0):
                continue
            hands.append({
                "hand_id": hand_idx,
                "side": side,
                "mano": {
                    "global_orient": pm[0:3].astype(np.float32),
                    "hand_pose":     pm[3:48].astype(np.float32),
                    "betas":         betas,
                    "translation":   pm[48:51].astype(np.float32),
                },
                "views": views_per_hand[hand_idx],
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
        frames_written += 1

        # Copy images.
        if not skip_images:
            for serial in serials:
                view_idx = serial_to_view[serial]
                src = os.path.join(seq_path, serial, f"color_{frame_idx:06d}.jpg")
                dst = os.path.join(
                    session_dir, "images", f"view{view_idx}", f"{frame_idx:06d}.jpg"
                )
                if os.path.exists(src):
                    shutil.copy2(src, dst)

    return frames_written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reformat DexYCB into multiview pinhole format."
    )
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the first 2 sessions.",
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Skip copying color JPEGs (faster for testing).",
    )
    args = parser.parse_args()

    calib_dir = os.path.join(args.data_dir, "calibration")
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect all sequences in subject → sorted-capture order.
    sequences = []   # list of (seq_path, subject_id)
    for subj in SUBJECTS:
        subj_dir = os.path.join(args.data_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        subj_id = subj.split("-subject-")[-1]   # '01' … '10'
        for seq in sorted(os.listdir(subj_dir)):
            seq_path = os.path.join(subj_dir, seq)
            if os.path.isdir(seq_path) and os.path.exists(
                os.path.join(seq_path, "meta.yml")
            ):
                sequences.append((seq_path, subj_id))

    if args.debug:
        sequences = sequences[:2]

    # Eval split: min(5%, 8) sessions, one per subject for diversity.
    n_total = len(sequences)
    n_eval = min(int(n_total * 0.05), 8)
    eval_indices = set()
    subjects_seen = set()
    # Walk sequences in order; take the first occurrence of each subject until
    # we have n_eval eval sessions.
    for i, (_, subj_id) in enumerate(sequences):
        if subj_id not in subjects_seen and len(eval_indices) < n_eval:
            eval_indices.add(i)
            subjects_seen.add(subj_id)

    train_dir = os.path.join(args.out_dir, "train")
    eval_dir = os.path.join(args.out_dir, "evaluation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Total sessions: {n_total}  |  eval: {len(eval_indices)}  |  train: {n_total - len(eval_indices)}")

    for idx, (seq_path, _) in enumerate(sequences):
        split = "evaluation" if idx in eval_indices else "train"
        name = session_name_from_path(seq_path)
        session_dir = os.path.join(args.out_dir, split, name)
        rel = os.path.relpath(seq_path, args.data_dir)
        print(f"[{idx+1:04d}/{n_total}] [{split[:5]}] {rel} → {name}", flush=True)
        n = process_sequence(seq_path, session_dir, calib_dir, args.skip_images)
        print(f"          {n} frames written", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()

"""
Verify a multi-view pinhole dataset by visualizing joints_2d and projected
joints_3d on sampled frames across views. Saves results to --out_dir.

Dataset layout:
  <data_root>/
    session_xxx/
      calib.npz          # array-indexed: K(N,3,3), dist(N,5),
                         #   R_world_to_cam(N,3,3), t_world_to_cam(N,3), img_size(N,2)
      frames/000001.npz  # single key "data": pickled dict with frame_id, hands list
      images/view0/000001.jpg
      ...

joints_3d in each view entry are assumed to be in that view's camera frame;
projection uses only K from calib.
"""

import os
import argparse
import random
import numpy as np
import cv2


# Joint indices in the stored array (original MANO ordering + tips):
#   wrist=0, thumb=13-15+16, index=1-3+17, middle=4-6+18, ring=10-12+19, pinky=7-9+20
SKELETON = [
    (0, 13), (13, 14), (14, 15), (15, 16),  # thumb
    (0,  1), ( 1,  2), ( 2,  3), ( 3, 17),  # index
    (0,  4), ( 4,  5), ( 5,  6), ( 6, 18),  # middle
    (0, 10), (10, 11), (11, 12), (12, 19),  # ring
    (0,  7), ( 7,  8), ( 8,  9), ( 9, 20),  # pinky
]

FINGER_COLORS = {
    "thumb":  (200, 0,   200),
    "index":  (255, 50,  50),
    "middle": (50,  255, 50),
    "ring":   (50,  50,  255),
    "pinky":  (255, 165, 0),
    "wrist":  (255, 255, 255),
}

JOINT_FINGER_MAP = {
    0:  "wrist",
    1:  "index",  2:  "index",  3:  "index",  17: "index",
    4:  "middle", 5:  "middle", 6:  "middle", 18: "middle",
    7:  "pinky",  8:  "pinky",  9:  "pinky",  20: "pinky",
    10: "ring",   11: "ring",   12: "ring",   19: "ring",
    13: "thumb",  14: "thumb",  15: "thumb",  16: "thumb",
}

BONE_FINGER = [
    "thumb",  "thumb",  "thumb",  "thumb",
    "index",  "index",  "index",  "index",
    "middle", "middle", "middle", "middle",
    "ring",   "ring",   "ring",   "ring",
    "pinky",  "pinky",  "pinky",  "pinky",
]

HAND_COLOR_SETS = [
    FINGER_COLORS,
    {k: (c[2], c[1], c[0]) for k, c in FINGER_COLORS.items()},
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def project_points(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N,3) camera-frame points to (N,2) pixel coords via K."""
    x = pts_cam[:, 0] / pts_cam[:, 2] * K[0, 0] + K[0, 2]
    y = pts_cam[:, 1] / pts_cam[:, 2] * K[1, 1] + K[1, 2]
    return np.stack([x, y], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _in_bounds(pt, w, h):
    return 0 <= pt[0] < w and 0 <= pt[1] < h


def draw_skeleton(img: np.ndarray, joints_2d: np.ndarray, color_map: dict) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for i, (a, b) in enumerate(SKELETON):
        color = color_map[BONE_FINGER[i]]
        pa = tuple(joints_2d[a].astype(int))
        pb = tuple(joints_2d[b].astype(int))
        if _in_bounds(pa, w, h) and _in_bounds(pb, w, h):
            cv2.line(out, pa, pb, color, 2, cv2.LINE_AA)
    for j, pt in enumerate(joints_2d):
        color = color_map[JOINT_FINGER_MAP[j]]
        pi = tuple(pt.astype(int))
        if _in_bounds(pi, w, h):
            cv2.circle(out, pi, 4, color, -1, cv2.LINE_AA)
            cv2.circle(out, pi, 4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_bbox(img: np.ndarray, bbox: np.ndarray, color, label: str):
    b = bbox.astype(int)
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 1)
    cv2.putText(img, label, (b[0], b[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def label_panel(img: np.ndarray, text: str):
    cv2.putText(img, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_calib(session_dir: str) -> dict:
    """
    Returns {view_idx(int): {"K", "dist", "R_world_to_cam", "t_world_to_cam", "img_size"}}.
    calib.npz stores each field as an array indexed by view: K(N,3,3), etc.
    """
    calib_path = os.path.join(session_dir, "calib.npz")
    if not os.path.isfile(calib_path):
        print(f"  MISSING calib.npz in {session_dir}")
        return {}

    raw = np.load(calib_path, allow_pickle=True)
    n_views = raw["K"].shape[0]
    calib = {}
    for vi in range(n_views):
        entry = {"K": raw["K"][vi]}
        for key in ("dist", "R_world_to_cam", "t_world_to_cam", "img_size"):
            if key in raw.files:
                entry[key] = raw[key][vi]
            else:
                print(f"  MISSING calib key '{key}'")
        calib[vi] = entry
    return calib


def load_frame(npz_path: str) -> dict:
    """
    Parse npz into: {"frame_id", "hands": [{"hand_id","side","mano","views":{vi:{...}}}]}.
    Supports a single pickled dict stored under key "data".
    """
    raw = np.load(npz_path, allow_pickle=True)

    # HO3D-style: everything in one pickled object under "data"
    if "data" in raw.files:
        return raw["data"].item()

    # Flat-key fallback (kept for forward compat)
    frame_id = int(raw["frame_id"])
    n_hands  = int(raw["n_hands"])
    hands = []
    for hi in range(n_hands):
        hp = f"hand{hi}_"
        side = str(raw[f"{hp}side"].item()) if f"{hp}side" in raw else "?"
        view_indices = raw[f"{hp}view_indices"].tolist() if f"{hp}view_indices" in raw else []
        views = {}
        for vi in view_indices:
            vp = f"{hp}view{vi}_"
            if f"{vp}joints_2d" not in raw:
                continue
            views[int(vi)] = {
                "bbox":      raw[f"{vp}bbox"],
                "joints_2d": raw[f"{vp}joints_2d"],
                "joints_3d": raw[f"{vp}joints_3d"],
            }
        mano = {mk: raw[f"{hp}mano_{mk}"] for mk in ("global_orient", "hand_pose", "betas")
                if f"{hp}mano_{mk}" in raw}
        hands.append({
            "hand_id": int(raw[f"{hp}hand_id"]) if f"{hp}hand_id" in raw else hi,
            "side": side, "mano": mano, "views": views,
        })
    return {"frame_id": frame_id, "hands": hands}


def verify_frame(frame: dict, npz_path: str) -> bool:
    ok = True
    for hi, hand in enumerate(frame["hands"]):
        for k in ("side", "mano", "views"):
            if k not in hand:
                print(f"  hand[{hi}] MISSING '{k}' in {npz_path}")
                ok = False
        for mk in ("global_orient", "hand_pose", "betas"):
            if mk not in hand.get("mano", {}):
                print(f"  hand[{hi}].mano MISSING '{mk}' in {npz_path}")
                ok = False
        for vi, vdata in hand.get("views", {}).items():
            for vk in ("bbox", "joints_2d", "joints_3d"):
                if vk not in vdata:
                    print(f"  hand[{hi}].view[{vi}] MISSING '{vk}' in {npz_path}")
                    ok = False
    return ok


def collect_frames(data_root: str, sessions: list) -> list:
    items = []
    for sess in sessions:
        frames_dir = os.path.join(data_root, sess, "frames")
        if not os.path.isdir(frames_dir):
            continue
        for fname in sorted(os.listdir(frames_dir)):
            if fname.endswith(".npz"):
                items.append((os.path.join(data_root, sess), fname[:-4]))
    return items


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_frame(session_dir: str, frame_stem: str,
                    calib: dict, out_path: str,
                    fallback_hw: tuple = (480, 640)):
    npz_path = os.path.join(session_dir, "frames", frame_stem + ".npz")
    frame = load_frame(npz_path)

    print(f"  frame_id={frame['frame_id']}  hands={len(frame['hands'])}", end="")
    for hand in frame["hands"]:
        print(f"  [{hand['side']} views={sorted(hand['views'].keys())}]", end="")
    print()

    if not verify_frame(frame, npz_path):
        print("  Key verification FAILED (continuing anyway)")

    all_view_indices = sorted({vi for hand in frame["hands"] for vi in hand["views"]})
    if not all_view_indices:
        print(f"  No view data — skipping")
        return

    rows = []
    for vi in all_view_indices:
        cam = calib.get(vi, {})
        K   = np.array(cam.get("K", np.eye(3)), dtype=np.float32)

        img_path = os.path.join(session_dir, "images", f"view{vi}", frame_stem + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            h, w = fallback_hw
            if "img_size" in cam:
                sz = cam["img_size"]
                w, h = int(sz[0]), int(sz[1])
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(img, f"missing: view{vi}/{frame_stem}.jpg",
                        (10, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (80, 80, 80), 1)

        panel_h, panel_w = img.shape[:2]
        left  = img.copy()
        right = img.copy()

        for hand_idx, hand in enumerate(frame["hands"]):
            if vi not in hand["views"]:
                continue
            vdata  = hand["views"][vi]
            colors = HAND_COLOR_SETS[hand_idx % 2]
            label  = "R" if hand["side"].lower() in ("right", "1") else "L"
            dot_color = (0, 128, 255) if label == "R" else (0, 255, 128)

            # Panel 1: stored joints_2d
            j2d = np.array(vdata["joints_2d"], dtype=np.float32)
            left = draw_skeleton(left, j2d, colors)
            draw_bbox(left, np.array(vdata["bbox"], dtype=np.float32), dot_color, label)

            # Panel 2: joints_3d (camera frame) projected via K
            j3d  = np.array(vdata["joints_3d"], dtype=np.float32)
            proj = project_points(j3d, K)
            right = draw_skeleton(right, proj, colors)
            draw_bbox(right, np.array(vdata["bbox"], dtype=np.float32), dot_color, label)

        label_panel(left,  f"view{vi}  joints_2d")
        label_panel(right, f"view{vi}  joints_3d proj")

        row = np.concatenate([left, right], axis=1)
        sep = np.full((3, row.shape[1], 3), 60, dtype=np.uint8)
        rows.extend([row, sep])

    rows = rows[:-1]
    max_w = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] == max_w:
            padded_rows.append(row)
            continue
        pad_w = max_w - row.shape[1]
        pad = np.zeros((row.shape[0], pad_w, 3), dtype=row.dtype)
        padded_rows.append(np.concatenate([row, pad], axis=1))

    composite = np.concatenate(padded_rows, axis=0)
    cv2.imwrite(out_path, composite)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify multi-view pinhole dataset")
    parser.add_argument("--data_root", type=str,
                        default="/lambda/nfs/hfm/qasim/hand_kp_dataset/dexycb/processed/multiview_pinhole/train")
    parser.add_argument("--n_frames",  type=int, default=5)
    parser.add_argument("--sessions",  type=str, default=None,
                        help="Comma-separated session names (default: all)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out_dir",   type=str, default="debug_imgs_multiview")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.sessions:
        sessions = [s.strip() for s in args.sessions.split(",")]
    else:
        sessions = sorted(
            s for s in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, s))
        )

    print(f"Sessions: {len(sessions)}")
    frames = collect_frames(args.data_root, sessions)
    print(f"Frames:   {len(frames)}")
    if not frames:
        print("No frames found. Exiting.")
        return

    sampled = random.sample(frames, min(args.n_frames, len(frames)))
    os.makedirs(args.out_dir, exist_ok=True)

    for session_dir, frame_stem in sampled:
        session_name = os.path.basename(session_dir)
        print(f"\nSession: {session_name}  frame: {frame_stem}")
        calib = load_calib(session_dir)
        out_path = os.path.join(args.out_dir, f"{session_name}__{frame_stem}.jpg")
        visualize_frame(session_dir, frame_stem, calib, out_path)


if __name__ == "__main__":
    main()

"""
Convert HO-3D v3 → the project's multi-view "session" format.

HO-3D ships per-camera *monocular* sequences plus a multi-camera calibration.
Multi-cam groups (e.g. ABF1) have 5 on-disk sequences ABF10..ABF14; the trailing
digit is the camera id, and all 5 record the SAME scene (shared frame ids), each
annotated in its own camera's OpenGL frame. We reconstruct true multi-view samples
from this. Single-cam sequences (MC*, ND2, SM*, SS*, SiS1, ...) → 1-view sessions.

Output layout (view0 = world frame):
    <out>/<split>/<session>/
        calib.npz   { K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3),
                      t_world_to_cam:(V,3), img_size:(V,2) }
        frames/NNNNNN.npz          # one logical multi-view sample (object array "data")
        images/view0/NNNNNN.jpg ... viewK/NNNNNN.jpg

Geometry (verified by reprojection — wrist maps to one world point from all views):
    flip = diag(1,-1,-1)                      # OpenGL -> OpenCV
    p_cv = flip @ p_gl
    For sequence <G>d: slot i = cam_orders.index(d); T_i = trans_i is cam->world (OpenCV):
        p_world = T_i @ [p_cv; 1]
    So per-view cam->world  M_k = T_{i_k} @ blkdiag(flip, 1).
    Rebase so view0 is world origin:  M'_k = inv(M_0) @ M_k ; world->cam = inv(M'_k).

frames/NNNNNN.npz "data" dict:
    { "frame_id": int, "is_annotated": bool,
      "hands": [ { "hand_id": 0, "side": "right",
                   "mano": {"global_orient":(3,), "hand_pose":(45,), "betas":(10,)},  # world frame
                   "views": { k: {"bbox":(4,), "joints_2d":(21,2), "joints_3d":(21,3)} } } ] }
HO-3D is right-hand only.
"""
import argparse
import glob
import os
import pickle
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

DEFAULT_HO3D_ROOT = "/lambda/nfs/hfm/qasim/hand_kp_dataset/ho3d"
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenGL -> OpenCV (applied to points)


# ── calibration parsing ───────────────────────────────────────────────────────

def load_intrinsics(path):
    """Parse cam_<i>_intrinsics.txt -> (K(3,3), dist(5,), (w,h))."""
    text = open(path).read().replace("\n", " ")
    fields = {}
    for part in text.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            fields[k.strip()] = v.strip()
    w, h = int(fields["width"]), int(fields["height"])
    fx, fy = float(fields["fx"]), float(fields["fy"])
    cx, cy = float(fields["ppx"]), float(fields["ppy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)  # HO-3D coeffs are all 0 (Brown-Conrady)
    return K, dist, (w, h)


def load_calib_group(calib_root, group):
    """Return dict slot_i -> (T_i 4x4) and the cam_orders list, plus intrinsics per cam id."""
    cdir = os.path.join(calib_root, group, "calibration")
    order = np.loadtxt(os.path.join(cdir, "cam_orders.txt")).astype(int).ravel().tolist()
    trans = {i: np.loadtxt(os.path.join(cdir, f"trans_{i}.txt")) for i in range(len(order))}
    intr = {}
    for cam_id in order:
        intr[cam_id] = load_intrinsics(os.path.join(cdir, f"cam_{cam_id}_intrinsics.txt"))
    return order, trans, intr


# ── session discovery ─────────────────────────────────────────────────────────

def discover_sessions(ho3d_root, split):
    """Return {session_name: {"group": str|None, "views": [(view_k, seq_dir, cam_id)]}}.

    Multi-cam: a calibration/<G>/ dir exists and seqs <G>0..<G>N are on disk.
    Single-cam: any remaining sequence becomes its own 1-view session.
    """
    split_dir = os.path.join(ho3d_root, split)
    calib_root = os.path.join(ho3d_root, "calibration")
    all_seqs = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    groups = sorted(os.listdir(calib_root)) if os.path.isdir(calib_root) else []

    sessions = {}
    claimed = set()
    for group in groups:
        order, _, _ = load_calib_group(calib_root, group)
        views = []
        for cam_id in order:  # view index follows cam_orders (slot order)
            seq = f"{group}{cam_id}"  # sequence digit == camera id
            seq_dir = os.path.join(split_dir, seq)
            if os.path.isdir(seq_dir):
                views.append((len(views), seq_dir, cam_id))
                claimed.add(seq)
        if views:
            sessions[group] = {"group": group, "views": views}

    for seq in all_seqs:
        if seq in claimed:
            continue
        sessions[seq] = {"group": None, "views": [(0, os.path.join(split_dir, seq), 0)]}
    return sessions


# ── per-session extrinsics ────────────────────────────────────────────────────

def build_calib(ho3d_root, session_info):
    """Return calib dict (K,dist,R_world_to_cam,t_world_to_cam,img_size) with view0=world."""
    calib_root = os.path.join(ho3d_root, "calibration")
    views = session_info["views"]
    V = len(views)
    K = np.zeros((V, 3, 3)); dist = np.zeros((V, 5))
    Rwc = np.zeros((V, 3, 3)); twc = np.zeros((V, 3)); img_size = np.zeros((V, 2), dtype=np.int64)

    if session_info["group"] is None:
        # single-cam: intrinsics from the frame's camMat (no calibration files)
        seq_dir = views[0][1]
        pkl = pickle.load(open(sorted(glob.glob(os.path.join(seq_dir, "meta", "*.pkl")))[0], "rb"))
        Kc = np.asarray(pkl["camMat"], dtype=np.float64)
        img = cv2.imread(sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.jpg")))[0])
        h, w = img.shape[:2]
        K[0] = Kc; img_size[0] = (w, h); Rwc[0] = np.eye(3)
        return {"K": K, "dist": dist, "R_world_to_cam": Rwc,
                "t_world_to_cam": twc, "img_size": img_size}

    order, trans, intr = load_calib_group(calib_root, session_info["group"])
    # trans_slot maps OpenCV cam points (p_cv = FLIP @ p_gl) directly to world, so
    # M_k (cam_cv->world) = trans[slot]; the FLIP is applied to the *points*, not folded here.
    M = []
    for k, (_, _, cam_id) in enumerate(views):
        slot = order.index(cam_id)
        M.append(trans[slot])
    M0_inv = np.linalg.inv(M[0])
    for k, (_, _, cam_id) in enumerate(views):
        Mk = M0_inv @ M[k]            # cam->world, rebased so view0 = origin
        w2c = np.linalg.inv(Mk)       # world->cam
        Rwc[k] = w2c[:3, :3]; twc[k] = w2c[:3, 3]
        Kc, dc, (w, h) = intr[cam_id]
        K[k] = Kc; dist[k] = dc; img_size[k] = (w, h)
    return {"K": K, "dist": dist, "R_world_to_cam": Rwc,
            "t_world_to_cam": twc, "img_size": img_size}


# ── per-frame conversion ──────────────────────────────────────────────────────

def project(K, pts_cv):
    """Project (N,3) OpenCV cam-frame points through K -> (N,2)."""
    uv = (K @ pts_cv.T).T
    return uv[:, :2] / uv[:, 2:3]


def bbox_from_2d(j2d, w, h, margin=0.15):
    x0, y0 = j2d.min(0); x1, y1 = j2d.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0 -= margin * bw; x1 += margin * bw; y0 -= margin * bh; y1 += margin * bh
    return np.array([max(0, x0), max(0, y0), min(w, x1), min(h, y1)], dtype=np.float32)


def load_meta(seq_dir, fid):
    path = os.path.join(seq_dir, "meta", f"{fid}.pkl")
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, "rb"))


def is_annotated(meta):
    return meta is not None and meta.get("handPose", None) is not None


def convert_frame(fid, views, calib):
    """Build the per-frame 'data' dict.

    Two HO-3D schemas are handled:
      * train  — full annotation: handPose(48), handBeta(10), handJoints3D(21,3).
                 Emits full mano + per-view joints_2d/joints_3d; is_annotated=True.
      * eval   — pose WITHHELD by the benchmark: only handBoundingBox + the 3-vec
                 root joint. Emits per-view bbox (+ root joint stored in joints_3d[0]),
                 mano zero-filled, joints_2d/joints_3d zero-filled, is_annotated=False
                 so withheld/absent pose is never used as supervision.
    """
    per_view = {}        # k -> view payload
    global_orient_world = None
    hand_pose = betas = None
    full_annotated = False     # any view has full train-style pose
    any_hand = False           # any view has at least a bbox (train or eval)

    for k, (_, seq_dir, _cam_id) in enumerate(views):
        meta = load_meta(seq_dir, fid)
        if meta is None:
            continue
        K = calib["K"][k]; w, h = calib["img_size"][k]

        if is_annotated(meta):
            # ── train: full pose ──────────────────────────────────────────────
            full_annotated = True
            j3d_gl = np.asarray(meta["handJoints3D"], dtype=np.float64)
            j3d_cv = (FLIP @ j3d_gl.T).T            # OpenCV cam frame for this view
            j2d = project(K, j3d_cv)
            bbox = bbox_from_2d(j2d, w, h)
            per_view[k] = {
                "bbox": bbox.astype(np.float32),
                "joints_2d": j2d.astype(np.float32),
                "joints_3d": j3d_cv.astype(np.float32),
            }
            # canonical MANO (view-independent pose/shape; global_orient rebased to world)
            if global_orient_world is None:
                pose = np.asarray(meta["handPose"], dtype=np.float64)
                hand_pose = pose[3:].astype(np.float32)
                betas = np.asarray(meta["handBeta"], dtype=np.float32)
                Rwc = calib["R_world_to_cam"][k]      # world->cam (OpenCV)
                R_cam_to_world = Rwc.T
                R_global_gl = R.from_rotvec(pose[:3]).as_matrix()
                R_global_cv = FLIP @ R_global_gl       # GL global-orient in OpenCV cam frame
                R_global_world = R_cam_to_world @ R_global_cv
                global_orient_world = R.from_matrix(R_global_world).as_rotvec().astype(np.float32)

        elif meta.get("handBoundingBox", None) is not None:
            # ── eval: withheld pose, only bbox + root joint ───────────────────
            bbox = np.asarray(meta["handBoundingBox"], dtype=np.float32)  # xyxy
            j3d = np.zeros((21, 3), dtype=np.float32)
            root_gl = np.asarray(meta["handJoints3D"], dtype=np.float64).reshape(-1)[:3]
            j3d[0] = (FLIP @ root_gl).astype(np.float32)   # root joint in OpenCV cam frame
            per_view[k] = {
                "bbox": bbox,
                "joints_2d": np.zeros((21, 2), dtype=np.float32),
                "joints_3d": j3d,                          # only joints_3d[0] (root) is valid
            }

        any_hand = any_hand or (k in per_view)

    hands = []
    if any_hand:
        if global_orient_world is None:   # eval-only frame: zero-fill mano
            global_orient_world = np.zeros(3, dtype=np.float32)
            hand_pose = np.zeros(45, dtype=np.float32)
            betas = np.zeros(10, dtype=np.float32)
        hands.append({
            "hand_id": 0, "side": "right",
            "mano": {"global_orient": global_orient_world,
                     "hand_pose": hand_pose, "betas": betas},
            "views": per_view,
        })
    # is_annotated marks full pose GT (train); eval frames are False so pose is never
    # used as supervision, but their bbox/root still ship under "hands".
    data = {"frame_id": int(fid), "is_annotated": bool(full_annotated), "hands": hands}
    return data


def frame_ids_for_session(views):
    """Frame ids present (as zero-padded strings) across views — union of rgb files."""
    ids = set()
    for _, seq_dir, _ in views:
        for p in glob.glob(os.path.join(seq_dir, "rgb", "*.jpg")):
            ids.add(os.path.splitext(os.path.basename(p))[0])
    return sorted(ids)


# ── verification ──────────────────────────────────────────────────────────────

def _wrist_cv(meta):
    """Return the wrist/root joint in OpenCV cam frame for either schema.

    train: handJoints3D is (21,3) -> joint 0. eval: handJoints3D is the (3,) root.
    """
    j = np.asarray(meta["handJoints3D"], dtype=np.float64)
    root_gl = j[0] if j.ndim == 2 else j.reshape(-1)[:3]
    return FLIP @ root_gl


def verify_session(views, calib, fids, n=3):
    """Assert multi-view world-point agreement of the wrist/root joint < 1mm.

    Works for both train (full joints) and eval (root-joint only) frames; the
    invariant — all views map the same physical joint to one world point — is the
    core correctness check on the extrinsics.
    """
    checked = 0
    for fid in fids:
        metas = [(k, load_meta(sd, fid)) for k, (_, sd, _) in enumerate(views)]
        ann = [(k, m) for k, m in metas
               if m is not None and (is_annotated(m) or m.get("handBoundingBox") is not None)]
        if len(ann) < 2:        # need ≥2 views to check agreement
            continue
        worlds = []
        for k, m in ann:
            p_cv = _wrist_cv(m)
            Rwc, twc = calib["R_world_to_cam"][k], calib["t_world_to_cam"][k]
            p_w = Rwc.T @ (p_cv - twc)      # world->cam: p_cam = Rwc p_w + twc
            worlds.append(p_w)
        spread = np.max(np.linalg.norm(np.array(worlds) - worlds[0], axis=1))
        assert spread < 1e-3, f"world spread {spread*1000:.2f}mm at frame {fid}"
        checked += 1
        if checked >= n:
            break
    return checked


# ── main ──────────────────────────────────────────────────────────────────────

def write_image(src, dst, symlink):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def session_is_complete(sess_dir, n_fids):
    """A session counts as already-converted if calib.npz exists and every frame
    .npz is present. Partial/interrupted sessions fail this and get redone."""
    if not os.path.exists(os.path.join(sess_dir, "calib.npz")):
        return False
    frames = glob.glob(os.path.join(sess_dir, "frames", "*.npz"))
    return len(frames) >= n_fids


def convert_session(name, info, ho3d_root, out_dir, symlink, verify, overwrite=False):
    views = info["views"]
    calib = build_calib(ho3d_root, info)
    fids = frame_ids_for_session(views)

    sess_dir = os.path.join(out_dir, name)
    if not overwrite and session_is_complete(sess_dir, len(fids)):
        return len(views), len(fids), None   # n_ann=None signals "skipped"
    os.makedirs(os.path.join(sess_dir, "frames"), exist_ok=True)
    np.savez(os.path.join(sess_dir, "calib.npz"), **calib)

    if verify:
        verify_session(views, calib, fids)

    n_ann = 0
    for fid in tqdm(fids, desc=name, leave=False):
        data = convert_frame(fid, views, calib)
        n_ann += int(data["is_annotated"])
        out_idx = f"{int(fid):06d}"
        np.savez(os.path.join(sess_dir, "frames", f"{out_idx}.npz"),
                 data=np.array(data, dtype=object))
        for k, (_, seq_dir, _) in enumerate(views):
            src = os.path.join(seq_dir, "rgb", f"{fid}.jpg")
            if os.path.exists(src):
                write_image(src, os.path.join(sess_dir, "images", f"view{k}", f"{out_idx}.jpg"), symlink)
    return len(views), len(fids), n_ann


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ho3d_root", default=DEFAULT_HO3D_ROOT)
    ap.add_argument("--out_root", default=os.path.join(DEFAULT_HO3D_ROOT, "ho3d_sessions"))
    ap.add_argument("--split", default="train", choices=["train", "evaluation", "all"],
                    help="'all' converts both train and evaluation in one run")
    ap.add_argument("--symlink-images", action="store_true", help="symlink instead of copying RGB")
    ap.add_argument("--limit", type=int, default=None, help="convert only first N sessions (smoke test)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-convert sessions even if already complete (default: skip/resume)")
    args = ap.parse_args()

    splits = ["train", "evaluation"] if args.split == "all" else [args.split]
    grand = {"sessions": 0, "skipped": 0, "frames": 0, "annotated": 0, "views": 0}

    for split in splits:
        sessions = discover_sessions(args.ho3d_root, split)
        names = sorted(sessions)
        if args.limit:
            names = names[: args.limit]
        out_dir = os.path.join(args.out_root, split)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{split}] Converting {len(names)} sessions → {out_dir}")
        for name in tqdm(names, desc=f"{split} sessions"):
            V, nf, na = convert_session(name, sessions[name], args.ho3d_root, out_dir,
                                        args.symlink_images, args.verify, args.overwrite)
            if na is None:   # already complete → skipped
                tqdm.write(f"  {name}: {V} views, {nf} frames — SKIPPED (already converted)")
                grand["skipped"] += 1
            else:
                tqdm.write(f"  {name}: {V} views, {na}/{nf} annotated frames")
                grand["annotated"] += na
            grand["sessions"] += 1; grand["frames"] += nf; grand["views"] += V

    print(f"\nDone. {grand['sessions']} sessions ({grand['skipped']} skipped), "
          f"{grand['frames']} frames ({grand['annotated']} annotated), "
          f"{grand['views']} total views → {args.out_root}")


if __name__ == "__main__":
    main()

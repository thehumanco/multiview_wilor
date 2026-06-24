"""
Convert H2O-3D → the project's multi-view "session" format.

H2O-3D is the two-hand sibling of HO-3D (same lab/tooling): each capture is
recorded by a STATIC multi-camera rig and annotated with BOTH hands (right +
left) plus the manipulated object, every hand/joint in that camera's own OpenGL
frame. On disk each camera is a separate sequence "<group><camid>" (camid 0..4);
synchronized cameras of a capture share frame ids. Unlike HO-3D there are NO
calibration files — but synchronized cameras observe the SAME physical points
each frame, so inter-camera extrinsics are recovered exactly (≈0 mm) by
Procrustes (Kabsch) on the shared hand joints + object corners.

NOT every "<group><camid>" sequence is part of the same synchronized capture:
a few cameras are unsynchronized recordings (e.g. SHB4 has 1023 frames vs 898
for SHB0-3 and fails to rigidly align). We therefore VALIDATE synchronization —
each candidate view must rigidly align with view0 below SYNC_THRESH_MM, else it
is split off into its own single-view session ("<group><camid>").

The official train/evaluation split holds out whole cameras (e.g. MBC0 → eval,
MBC1..MBC4 → train), so a group can appear as a 4-view session in train and a
1-view session in eval. We build one multi-view session per synchronized
(group, split) cluster.

CONSISTENT VIEW INDEXING: the rig has exactly 5 physical cameras, identified by
the trailing camid (0..4) of each "<group><camid>" sequence — camid is a true
physical-camera id (each camid has its own distinct intrinsics, stable across
groups). We therefore index views by camid: view k IS physical camera k in
EVERY multi-view session. Absent cameras (held out to the other split) leave a
zero-filled gap slot that the session dataloader emits as an empty placeholder.
This makes a view index mean the same camera everywhere, so a globally-consistent
BAD_VIEWS list (e.g. a camera whose hands are consistently occluded) is valid
across all sessions/splits. (Naive enumerate-by-position indexing would make
train view0=cam1 but eval view0=cam0 — the bug this fixes.) The world frame is
the lowest-camid synced camera (its slot is identity); single-view sessions are
stored at slot 0 (a lone view has no cross-view occlusion to compare).

H2O-3D has NO egocentric view — all 5 cameras are static third-person (the
egocentric "H2O" dataset is a different, similarly-named capture).

Output layout (view k == physical camera k; world == lowest synced camid):
    <out>/<split>/<session>/
        calib.npz   { K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3),
                      t_world_to_cam:(V,3), img_size:(V,2) }   # V = max camid+1
        frames/NNNNNN.npz          # one logical multi-view sample (object array "data")
        images/view0/NNNNNN.jpg ... viewK/NNNNNN.jpg          # gap slots have no images

Geometry (verified by reprojection + multi-view world agreement):
    flip = diag(1,-1,-1)                      # OpenGL -> OpenCV (applied to points)
    p_cv = flip @ p_gl
    world == the world-slot camera's (OpenCV) frame. For view k,
    (R_world_to_cam[k], t_world_to_cam[k]) is the rigid transform mapping a world
    point to view k's cv frame, found by Kabsch on object corners. The world slot
    is identity. global_orient is rebased to world via R_cam_to_world @ (flip @ R_gl).

frames/NNNNNN.npz "data" dict:
    { "frame_id": int, "is_annotated": bool,
      "hands": [ { "hand_id": 0|1, "side": "right"|"left",
                   "mano": {"global_orient":(3,), "hand_pose":(45,), "betas":(10,)},  # world frame
                   "views": { k: {"bbox":(4,), "joints_2d":(21,2), "joints_3d":(21,3)} } } ] }
                 # view key k == physical camera id (camid)

Two HO-3D-style schemas are handled per frame:
  * train — full pose: {right,left}HandPose(48), handBeta(10),
            {right,left}HandJoints3D(21,3), {joint,pose}Valid{Right,Left}.
            Emits full mano + per-view joints_2d/joints_3d; is_annotated=True.
  * eval  — pose WITHHELD: only {right,left}HandBoundingBox + the (3,) root joint
            in {right,left}HandJoints3D. Emits per-view bbox (+ root in
            joints_3d[0]), mano zero-filled, is_annotated=False.

H2O-3D handJoints3D use the same 21-joint order as HO-3D (stored as-is; the
session dataloader remaps to OpenPose).
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

DEFAULT_H2O3D_ROOT = "/lambda/nfs/hfm/qasim/hand_kp_dataset/h2o3d"
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenGL -> OpenCV (applied to points)

# A hand counts as "seen" by a view when at least this many of its valid joints
# project inside the image; cameras not actually looking at the hand are dropped.
MIN_VIS_JOINTS = 6
# Frames sampled per session to fit the (static) inter-camera extrinsics.
N_CALIB_FRAMES = 60
# A view is part of the synchronized rig only if its Procrustes fit to view0 is
# below this (mm); unsynchronized cameras score tens of mm and are split off.
SYNC_THRESH_MM = 5.0
MIN_CORR_POINTS = 12  # minimum correspondences needed to trust an extrinsic fit

# (side string, capitalized suffix used by *Valid* keys, hand_id)
HAND_DEFS = [("right", "Right", 0), ("left", "Left", 1)]


# ── meta IO ─────────────────────────────────────────────────────────────────--

def load_meta(seq_dir, fid):
    """Load <seq>/meta/<fid>.pkl, or None if missing/corrupt.

    A few H2O-3D pickles fail to unpickle under newer numpy ("_reconstruct:
    First argument must be a sub-type of ndarray"); those frames are treated as
    absent for that view rather than aborting the session.
    """
    path = os.path.join(seq_dir, "meta", f"{fid}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def is_train_meta(meta):
    return meta is not None and "rightHandPose" in meta


# ── session discovery ─────────────────────────────────────────────────────────

def discover_sessions(h2o3d_root, split):
    """Return {session_name: {"views": [(slot, seq_dir, camid)]}}.

    Sequence "<group><camid>" → group = name[:-1], camid = int(name[-1]).
    All sequences sharing a group within this split become one multi-view
    session. The view slot is the camid itself (view k == physical camera k),
    so the view↔camera mapping is identical across groups and splits. Names
    that don't end in a 0-4 digit fall back to their own single-view session
    (camid 0).
    """
    split_dir = os.path.join(h2o3d_root, split)
    seqs = sorted(d for d in os.listdir(split_dir)
                  if os.path.isdir(os.path.join(split_dir, d)))

    groups = {}  # group_name -> list of (camid, seq_dir)
    for seq in seqs:
        seq_dir = os.path.join(split_dir, seq)
        if seq[-1:].isdigit() and len(seq) > 1:
            group, camid = seq[:-1], int(seq[-1])
        else:
            group, camid = seq, 0
        groups.setdefault(group, []).append((camid, seq_dir))

    sessions = {}
    for group, members in groups.items():
        members.sort(key=lambda x: x[0])  # order views by camid; world = smallest
        views = [(camid, seq_dir, camid) for camid, seq_dir in members]
        sessions[group] = {"views": views}
    return sessions


def frame_ids_for_session(views):
    """Zero-padded frame-id strings present across views (union of rgb files)."""
    ids = set()
    for _, seq_dir, _ in views:
        for p in glob.glob(os.path.join(seq_dir, "rgb", "*.jpg")):
            ids.add(os.path.splitext(os.path.basename(p))[0])
    return sorted(ids)


# ── geometry ───────────────────────────────────────────────────────────────--

def kabsch(A, B):
    """Rigid transform (R, t) mapping points A (N,3) onto B (N,3): B ≈ A @ R.T + t."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Rm, cB - Rm @ cA


def project(K, pts_cv):
    """Project (N,3) OpenCV cam-frame points through K -> (N,2)."""
    uv = (K @ pts_cv.T).T
    return uv[:, :2] / uv[:, 2:3]


def bbox_from_2d(j2d, w, h, margin=0.15):
    x0, y0 = j2d.min(0); x1, y1 = j2d.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0 -= margin * bw; x1 += margin * bw; y0 -= margin * bh; y1 += margin * bh
    return np.array([max(0, x0), max(0, y0), min(w, x1), min(h, y1)], dtype=np.float32)


# ── correspondences & inter-camera fit ────────────────────────────────────────

def corr_points(meta):
    """Map {point_id -> (3,) OpenCV-frame coord} of every reliably-annotated
    physical point in this meta: object corners ('O',0..7), plus hand joints
    ('R'/'L', j) (train, jointValid only) or hand roots ('R'/'L', 0) (eval).

    These ids are consistent across cameras of the same frame, so intersecting
    two views' dicts yields rigid-body correspondences.
    """
    pts = {}
    obj = np.asarray(meta.get("objCorners3D"), dtype=np.float64) if "objCorners3D" in meta else None
    if obj is not None and obj.shape == (8, 3) and np.isfinite(obj).all():
        for i, p in enumerate((FLIP @ obj.T).T):
            pts[("O", i)] = p
    for side, Side, _ in HAND_DEFS:
        j = np.asarray(meta[f"{side}HandJoints3D"], dtype=np.float64)
        if is_train_meta(meta):
            jv = np.asarray(meta[f"jointValid{Side}"]).astype(bool)
            jcv = (FLIP @ j.T).T
            for i in np.nonzero(jv)[0]:
                pts[(side, int(i))] = jcv[i]
        else:  # eval: only the (3,) root joint is provided
            root = j.reshape(-1)[:3]
            if np.isfinite(root).all():
                pts[(side, 0)] = FLIP @ root
    return pts


def gather_correspondences(ref_dir, other_dir, fids):
    """Accumulate (A, B) point pairs over sampled frames where ref and other
    share annotated physical points. A is in ref frame, B in other frame."""
    sample = fids[:: max(1, len(fids) // N_CALIB_FRAMES)][:N_CALIB_FRAMES]
    A, B = [], []
    for fid in sample:
        m0, mk = load_meta(ref_dir, fid), load_meta(other_dir, fid)
        if m0 is None or mk is None:
            continue
        p0, pk = corr_points(m0), corr_points(mk)
        for key in p0.keys() & pk.keys():
            A.append(p0[key]); B.append(pk[key])
    if not A:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.array(A), np.array(B)


def fit_view_to_ref(ref_dir, other_dir, fids):
    """Return (R_world_to_cam, t_world_to_cam, residual_mm, n_points) for `other`
    relative to `ref` (= world). residual is inf if too few correspondences."""
    A, B = gather_correspondences(ref_dir, other_dir, fids)
    if len(A) < MIN_CORR_POINTS:
        return None, None, float("inf"), len(A)
    Rwc, twc = kabsch(A, B)               # maps world(ref) point -> other cam
    res = np.linalg.norm((Rwc @ A.T).T + twc - B, axis=1).mean() * 1000.0
    return Rwc, twc, res, len(A)


# ── calibration (intrinsics from camMat, extrinsics via Procrustes) ───────────--

def _read_intrinsics(seq_dir, fids):
    meta = next((m for m in (load_meta(seq_dir, f) for f in fids) if m is not None), None)
    if meta is None:
        raise RuntimeError(f"no readable meta in {seq_dir}")
    K = np.asarray(meta["camMat"], dtype=np.float64)
    img = cv2.imread(sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.jpg")))[0])
    h, w = img.shape[:2]
    return K, np.array([w, h], dtype=np.int64)


def build_calib(views, fits, world_slot):
    """calib dict (K,dist,R_world_to_cam,t_world_to_cam,img_size) indexed by slot.

    `views` are the already-synchronized cluster as (slot, seq_dir, camid) with
    slot == camid; `fits` maps slot (!= world_slot) -> (Rwc, twc) from
    fit_view_to_ref. Arrays are sized to max(slot)+1 so view k == camera k; any
    camid absent from this cluster is a zero/identity-filled gap slot (the
    dataloader only ever yields it as an empty placeholder). Intrinsics come
    from each camera's `camMat`; the world-slot extrinsics are identity.
    """
    V = max(slot for slot, _, _ in views) + 1
    present = {slot for slot, _, _ in views}
    K = np.zeros((V, 3, 3)); dist = np.zeros((V, 5))
    Rwc = np.tile(np.eye(3), (V, 1, 1)); twc = np.zeros((V, 3))
    img_size = np.zeros((V, 2), dtype=np.int64)
    for slot, seq_dir, camid in views:
        K[slot], img_size[slot] = _read_intrinsics(
            seq_dir, frame_ids_for_session([(slot, seq_dir, camid)]))
        if slot != world_slot:
            Rwc[slot], twc[slot] = fits[slot]
    # gap slots (absent cameras): keep identity extrinsics, borrow world's K/size
    # so the placeholder samples carry sane (unused) intrinsics.
    for slot in range(V):
        if slot not in present:
            K[slot] = K[world_slot]; img_size[slot] = img_size[world_slot]
    return {"K": K, "dist": dist, "R_world_to_cam": Rwc,
            "t_world_to_cam": twc, "img_size": img_size}


def partition_synced(views):
    """Split a prefix group's candidate views into a synchronized cluster (the
    lowest-camid reference + every view that rigidly aligns to it) and the
    leftover unsynced views. Slots stay equal to camid. Returns
    (synced_views, fits, world_slot, unsynced_views) where fits[slot] = (Rwc,
    twc) for slot != world_slot.
    """
    world_slot, ref_dir, _ = views[0]   # lowest camid = world reference
    ref_fids = frame_ids_for_session([views[0]])
    synced = [views[0]]
    fits = {}
    unsynced = []
    for slot, seq_dir, camid in views[1:]:
        Rwc, twc, res, n = fit_view_to_ref(ref_dir, seq_dir, ref_fids)
        if res < SYNC_THRESH_MM:
            synced.append((slot, seq_dir, camid))
            fits[slot] = (Rwc, twc)
        else:
            unsynced.append((0, seq_dir, camid))
    return synced, fits, world_slot, unsynced


# ── per-frame conversion ──────────────────────────────────────────────────────

def convert_frame(fid, views, calib):
    """Build the per-frame 'data' dict for both hands."""
    per_view = {0: {}, 1: {}}        # hand_id -> {view_k: payload}
    go_world = {0: None, 1: None}    # hand_id -> global_orient (world axis-angle)
    hand_pose = {0: None, 1: None}
    betas = {0: None, 1: None}
    full_annotated = False

    for slot, seq_dir, _camid in views:
        meta = load_meta(seq_dir, fid)
        if meta is None:
            continue
        Kk = calib["K"][slot]; w, h = calib["img_size"][slot]

        for side, Side, hid in HAND_DEFS:
            if is_train_meta(meta):
                # ── train: full pose ──────────────────────────────────────────
                jv = np.asarray(meta[f"jointValid{Side}"]).astype(bool)
                if not jv.any():
                    continue
                j3d_gl = np.asarray(meta[f"{side}HandJoints3D"], dtype=np.float64)  # (21,3)
                j3d_cv = (FLIP @ j3d_gl.T).T
                j2d = project(Kk, j3d_cv)
                inb = ((j2d[:, 0] >= 0) & (j2d[:, 0] < w) &
                       (j2d[:, 1] >= 0) & (j2d[:, 1] < h))
                use = jv & inb
                if use.sum() < MIN_VIS_JOINTS:
                    continue  # this camera doesn't really see the hand
                per_view[hid][slot] = {
                    "bbox": bbox_from_2d(j2d[use], w, h).astype(np.float32),
                    "joints_2d": j2d.astype(np.float32),
                    "joints_3d": j3d_cv.astype(np.float32),
                }
                full_annotated = True
                # canonical MANO (pose/shape view-independent; global_orient -> world)
                if go_world[hid] is None and np.asarray(meta[f"poseValid{Side}"]).any():
                    pose = np.asarray(meta[f"{side}HandPose"], dtype=np.float64)  # (48,)
                    hand_pose[hid] = pose[3:].astype(np.float32)
                    betas[hid] = np.asarray(meta["handBeta"], dtype=np.float32)
                    R_cam_to_world = calib["R_world_to_cam"][slot].T
                    R_global_cv = FLIP @ R.from_rotvec(pose[:3]).as_matrix()
                    go_world[hid] = R.from_matrix(
                        R_cam_to_world @ R_global_cv).as_rotvec().astype(np.float32)

            elif meta.get(f"{side}HandBoundingBox", None) is not None:
                # ── eval: withheld pose, only bbox + root joint ───────────────
                bbox = np.asarray(meta[f"{side}HandBoundingBox"], dtype=np.float32)  # xyxy
                j3d = np.zeros((21, 3), dtype=np.float32)
                root_gl = np.asarray(meta[f"{side}HandJoints3D"], dtype=np.float64).reshape(-1)[:3]
                j3d[0] = (FLIP @ root_gl).astype(np.float32)
                per_view[hid][slot] = {
                    "bbox": bbox,
                    "joints_2d": np.zeros((21, 2), dtype=np.float32),
                    "joints_3d": j3d,  # only joints_3d[0] (root) is valid
                }

    hands = []
    for side, _Side, hid in HAND_DEFS:
        if not per_view[hid]:
            continue
        if go_world[hid] is None:  # eval-only hand: zero-fill mano
            go_world[hid] = np.zeros(3, dtype=np.float32)
            hand_pose[hid] = np.zeros(45, dtype=np.float32)
            betas[hid] = np.zeros(10, dtype=np.float32)
        hands.append({
            "hand_id": hid, "side": side,
            "mano": {"global_orient": go_world[hid],
                     "hand_pose": hand_pose[hid], "betas": betas[hid]},
            "views": per_view[hid],
        })
    return {"frame_id": int(fid), "is_annotated": bool(full_annotated), "hands": hands}


# ── verification ──────────────────────────────────────────────────────────────

def verify_session(views, calib, fids, n=3):
    """Assert that, for hands seen by ≥2 views, every view maps the wrist/root to
    one world point (< 1 mm spread) — the core extrinsics-correctness check."""
    checked = 0
    for fid in fids:
        data = convert_frame(fid, views, calib)
        verified_any = False
        for hand in data["hands"]:
            vlist = list(hand["views"].items())
            if len(vlist) < 2:
                continue
            worlds = []
            for k, vw in vlist:
                p_cam = np.asarray(vw["joints_3d"][0], dtype=np.float64)  # wrist/root, cv
                Rwc, twc = calib["R_world_to_cam"][k], calib["t_world_to_cam"][k]
                worlds.append(Rwc.T @ (p_cam - twc))
            spread = np.max(np.linalg.norm(np.array(worlds) - worlds[0], axis=1))
            assert spread < 1e-3, f"world spread {spread*1000:.2f}mm at frame {fid}"
            verified_any = True
        if verified_any:
            checked += 1
            if checked >= n:
                break
    return checked


# ── session driver ────────────────────────────────────────────────────────────

def write_image(src, dst, symlink):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def session_is_complete(sess_dir, n_fids):
    if not os.path.exists(os.path.join(sess_dir, "calib.npz")):
        return False
    return len(glob.glob(os.path.join(sess_dir, "frames", "*.npz"))) >= n_fids


def _write_session(name, views, calib, out_dir, symlink, verify, overwrite):
    """Write one session (calib + per-frame npz + images) for an ordered,
    reindexed view list. Returns (n_views, n_frames, n_annotated | None-if-skipped)."""
    fids = frame_ids_for_session(views)
    sess_dir = os.path.join(out_dir, name)
    if not overwrite and session_is_complete(sess_dir, len(fids)):
        return len(views), len(fids), None  # n_ann=None signals "skipped"

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
        for slot, seq_dir, _camid in views:
            src = os.path.join(seq_dir, "rgb", f"{fid}.jpg")
            if os.path.exists(src):
                write_image(src, os.path.join(
                    sess_dir, "images", f"view{slot}", f"{out_idx}.jpg"), symlink)
    return len(views), len(fids), n_ann


def convert_group(name, info, out_dir, symlink, verify, overwrite=False):
    """Convert one prefix group, splitting it into a synchronized multi-view
    session ("<group>") plus a single-view session ("<group><camid>") for each
    unsynchronized leftover camera. Returns a list of (session_name, result).
    """
    views = info["views"]
    if len(views) == 1:
        solo = [(0, views[0][1], views[0][2])]   # lone view -> slot 0
        calib = build_calib(solo, {}, world_slot=0)
        return [(name, _write_session(name, solo, calib, out_dir, symlink, verify, overwrite))]

    synced, fits, world_slot, unsynced = partition_synced(views)
    results = []
    if len(synced) == 1:
        # only the reference aligned -> no multi-view cluster; store it as solo
        solo = [(0, synced[0][1], synced[0][2])]
        calib = build_calib(solo, {}, world_slot=0)
        results.append((name, _write_session(name, solo, calib, out_dir, symlink, verify, overwrite)))
    else:
        calib = build_calib(synced, fits, world_slot)
        results.append((name, _write_session(name, synced, calib, out_dir, symlink, verify, overwrite)))
    for v in unsynced:
        seq_name = os.path.basename(v[1])  # original "<group><camid>"
        solo = [(0, v[1], v[2])]
        scalib = build_calib(solo, {}, world_slot=0)
        results.append((seq_name,
                        _write_session(seq_name, solo, scalib, out_dir, symlink, verify, overwrite)))
    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2o3d_root", default=DEFAULT_H2O3D_ROOT)
    ap.add_argument("--out_root", default=os.path.join(DEFAULT_H2O3D_ROOT, "sessions"))
    ap.add_argument("--split", default="all", choices=["train", "evaluation", "all"])
    ap.add_argument("--symlink-images", action="store_true",
                    help="symlink instead of copying RGB")
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only first N sessions (smoke test)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-convert complete sessions (default: skip/resume)")
    args = ap.parse_args()

    splits = ["train", "evaluation"] if args.split == "all" else [args.split]
    grand = {"sessions": 0, "skipped": 0, "frames": 0, "annotated": 0, "views": 0}

    for split in splits:
        sessions = discover_sessions(args.h2o3d_root, split)
        names = sorted(sessions)
        if args.limit:
            names = names[: args.limit]
        out_dir = os.path.join(args.out_root, split)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{split}] Converting {len(names)} groups → {out_dir}")
        for name in tqdm(names, desc=f"{split} groups"):
            for sname, (V, nf, na) in convert_group(
                    name, sessions[name], out_dir,
                    args.symlink_images, args.verify, args.overwrite):
                if na is None:
                    tqdm.write(f"  {sname}: {V} views, {nf} frames — SKIPPED (already converted)")
                    grand["skipped"] += 1
                else:
                    tqdm.write(f"  {sname}: {V} views, {na}/{nf} annotated frames")
                    grand["annotated"] += na
                grand["sessions"] += 1; grand["frames"] += nf; grand["views"] += V

    print(f"\nDone. {grand['sessions']} sessions ({grand['skipped']} skipped), "
          f"{grand['frames']} frames ({grand['annotated']} annotated), "
          f"{grand['views']} total views → {args.out_root}")


if __name__ == "__main__":
    main()

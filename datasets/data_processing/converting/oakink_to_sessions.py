#!/usr/bin/env python3
"""
Convert OakInk-v2 (anno_preview pickles + extracted PNGs) → the project's
multi-view "session" format.

OakInk-v2 records each sequence with 4 cameras (allocentric top/left/right +
egocentric) and ships per-frame SMPL-X/MANO mocap annotations. The 3 allocentric
cameras form a STATIC rig and define the world frame (view0 = allocentric_top);
the egocentric camera is head-mounted and MOVES (its cam_extr varies frame-to-frame
while its intrinsics are static), so it is emitted as a DYNAMIC view (index 3): its
per-frame world->cam is stored in each frame npz, and calib.npz flags it via
dynamic_views. We emit one 4-view session per sequence.

Output layout (view0 = world frame):
    <out>/<session>/
        calib.npz   { K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3),
                      t_world_to_cam:(V,3), img_size:(V,2),
                      dynamic_views:(1,)=[3] }   # view 3 (ego) R/t = frame-0 placeholder
        frames/NNNNNN.npz          # one logical multi-view sample (object array "data")
        images/view0/NNNNNN.jpg ... view3/NNNNNN.jpg   # 3 allocentric + 1 ego view

frames/NNNNNN.npz "data" dict:
    { "frame_id": int, "is_annotated": True,
      "hands": [ { "hand_id": 0|1, "side": "right"|"left",
                   "mano": {"global_orient":(3,), "hand_pose":(45,), "betas":(10,)},  # world frame, axis-angle
                   # only the views that actually see this hand:
                   "views": { k: {"bbox":(4,), "joints_2d":(21,2), "joints_3d":(21,3),
                                  # ego view (3) only: per-frame world->cam [R|t]
                                  ["extrinsics":(3,4)]} } } ] }

Verified OakInk conventions (see OakInk_to_intermediate_format_conversion.md):
  * cam_extr[name][fid] is world->cam (4x4); cam_intr[name][fid] is K (3x3).
    Both are STATIC across frames within a sequence.
  * image frame_id indexes raw_mano / cam_intr / cam_extr directly (== mocap idx).
  * raw_mano[fid]["{rh,lh}__pose_coeffs"] is (1,16,4) quaternion (wxyz): joint 0
    is global_orient, joints 1..15 are the hand pose. "__tsl" (1,3) is the wrist
    world translation; "__betas" (1,10) shape. global_orient is already world-frame.
  * MANO world joints = (root-relative 21 joints) + tsl.

21-joint layout matches HO-3D handJoints3D: smplx MANO's 16 joints +
5 fingertip vertices in thumb/index/middle/ring/pinky order.
"""
import argparse
import contextlib
import glob
import logging
import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import inspect
import warnings 

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# repo MANO weights live here; smplx wants a "mano/MANO_*.pkl" subdir layout
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
DEFAULT_OAKINK_ROOT = "/lambda/nfs/hfm/qasim/hand_kp_dataset/oakink"
DEFAULT_MANO_PKL_DIR = os.path.join(_REPO_ROOT, "src/metric_hand_tracking/wilor/mano_data")

# view0 = world frame. The 3 STATIC allocentric cameras form the rig and define the world
# frame (view0 = allocentric_top). The egocentric camera is head-mounted and MOVES across
# frames (its cam_extr varies frame-to-frame while its intrinsics are static), so it is added
# as a DYNAMIC view (index EGO_VIEW = 3): its per-frame world->cam [R|t] is written into each
# frame npz under views[3]["extrinsics"], and calib.npz holds only the static ego intrinsics +
# a frame-0 placeholder R/t. Fixed ordering below.
VIEW_ORDER = ["allocentric_top", "allocentric_left", "allocentric_right"]
EGO_NAME = "egocentric"
EGO_VIEW = len(VIEW_ORDER)  # dynamic ego view index (== 3)
IMG_W, IMG_H = 848, 480

# HO-3D handJoints3D order is MANO's 16 joints plus fingertips:
# wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3, then
# thumb/index/middle/ring/pinky tips. smplx MANO joints already provide the
# first 16 entries in this order; append fingertips without OpenPose reordering.
FINGERTIP_VERTS = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671}


# ── MANO ──────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_smplx_output():
    """Silence smplx's "10 shape coefficients" notice + any stdout/stderr it emits
    on model creation, so it can't smear the tqdm bar (esp. across workers)."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        smplx_logger = logging.getLogger("smplx")
        old_level = smplx_logger.level
        try:
            sys.stdout = sys.stderr = devnull
            smplx_logger.setLevel(logging.CRITICAL)
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            smplx_logger.setLevel(old_level)


def setup_mano_dir(mano_pkl_dir):
    """smplx.create needs <dir>/mano/MANO_{LEFT,RIGHT}.pkl. If mano_pkl_dir holds
    the bare pkls, build a temp parent with a `mano/` subdir of symlinks and
    return that parent."""
    if os.path.basename(mano_pkl_dir.rstrip("/")) == "mano":
        return os.path.dirname(mano_pkl_dir.rstrip("/"))
    import tempfile
    parent = tempfile.mkdtemp(prefix="oakink_mano_")
    sub = os.path.join(parent, "mano")
    os.makedirs(sub, exist_ok=True)
    for f in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
        src = os.path.join(mano_pkl_dir, f)
        if os.path.exists(src):
            os.symlink(src, os.path.join(sub, f))
    return parent


def build_mano_models(mano_parent, device):
    import smplx
    models = {}
    with _suppress_smplx_output():
        for side, is_rhand in (("right", True), ("left", False)):
            m = smplx.create(
                model_path=mano_parent, model_type="mano", use_pca=False,
                is_rhand=is_rhand, flat_hand_mean=True, batch_size=1,
            ).to(device).eval()
            models[side] = m
    return models


def mano_world_joints(model, pose_coeffs, betas, tsl, device):
    """OakInk MANO params -> (21,3) world joints + (3,) global_orient axis-angle.

    pose_coeffs: (16,4) quaternion (wxyz). betas: (10,). tsl: (3,).
    """
    quat_xyzw = pose_coeffs[:, [1, 2, 3, 0]]              # wxyz -> xyzw
    aa = R.from_quat(quat_xyzw).as_rotvec()               # (16,3)
    global_orient = aa[0].astype(np.float32)
    hand_pose = aa[1:].reshape(-1).astype(np.float32)     # (45,)
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(global_orient[None], device=device),
            hand_pose=torch.tensor(hand_pose[None], device=device),
            betas=torch.tensor(betas[None].astype(np.float32), device=device),
        )
    j16 = out.joints[0].cpu().numpy()                     # (16,3)
    verts = out.vertices[0].cpu().numpy()                 # (778,3)
    tips = np.stack([verts[FINGERTIP_VERTS[f]] for f in
                     ["thumb", "index", "middle", "ring", "pinky"]], 0)  # (5,3)
    j21 = np.concatenate([j16, tips], 0)                                 # (21,3)
    j_world = (j21 - j21[0] + tsl).astype(np.float32)
    return j_world, global_orient


# ── calibration ───────────────────────────────────────────────────────────────

def build_calib(anno):
    """calib dict (K,dist,R_world_to_cam,t_world_to_cam,img_size,dynamic_views) with view0=world.

    The 3 static allocentric views (0..2) plus the dynamic egocentric view (EGO_VIEW=3) are
    stacked. The ego entry stores static intrinsics + a frame-0 placeholder R/t (the real
    per-frame ego world->cam lives in each frame npz); dynamic_views=[EGO_VIEW] flags it.

    Also returns world_to_v0 (4x4) = E_0 (original-world -> cam0), used to rebase MANO world
    points (and the per-frame ego extrinsic) into the view0 world frame (new world == cam0).
    """
    f0 = anno["frame_id_list"][0]
    V = len(VIEW_ORDER) + 1  # static allocentric views + ego
    K = np.zeros((V, 3, 3)); dist = np.zeros((V, 5))
    Rwc = np.zeros((V, 3, 3)); twc = np.zeros((V, 3))
    img_size = np.zeros((V, 2), dtype=np.int64)

    E = [np.asarray(anno["cam_extr"][name][f0], dtype=np.float64) for name in VIEW_ORDER]  # world->cam
    # New world frame = camera-0 frame. world->cam_k (rebased) = E_k @ inv(E_0)
    E0_inv = np.linalg.inv(E[0])
    for k, name in enumerate(VIEW_ORDER):
        w2c = E[k] @ E0_inv
        Rwc[k] = w2c[:3, :3]; twc[k] = w2c[:3, 3]
        K[k] = np.asarray(anno["cam_intr"][name][f0], dtype=np.float64)
        img_size[k] = (IMG_W, IMG_H)
    # ego view: static K, frame-0 placeholder extrinsics (rebased), ego image size.
    E_ego0 = np.asarray(anno["cam_extr"][EGO_NAME][f0], dtype=np.float64) @ E0_inv
    Rwc[EGO_VIEW] = E_ego0[:3, :3]; twc[EGO_VIEW] = E_ego0[:3, 3]
    K[EGO_VIEW] = np.asarray(anno["cam_intr"][EGO_NAME][f0], dtype=np.float64)
    img_size[EGO_VIEW] = (IMG_W, IMG_H)
    calib = {"K": K, "dist": dist, "R_world_to_cam": Rwc,
             "t_world_to_cam": twc, "img_size": img_size,
             "dynamic_views": np.array([EGO_VIEW], dtype=np.int64)}
    # new world frame == cam0 frame, so original-world -> new-world is E_0 itself.
    return calib, E[0]


# ── per-frame conversion ──────────────────────────────────────────────────────

def project(K, pts_cam):
    """Project (N,3) camera-frame points through K -> (N,2)."""
    uv = (K @ pts_cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def bbox_from_2d(j2d, w, h, margin=0.15):
    x0, y0 = j2d.min(0); x1, y1 = j2d.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0 -= margin * bw; x1 += margin * bw; y0 -= margin * bh; y1 += margin * bh
    return np.array([max(0, x0), max(0, y0), min(w, x1), min(h, y1)], dtype=np.float32)


def view_sees_hand(j2d, z, w, h):
    """A view sees the hand if all joints are in front of the camera and a
    meaningful fraction of joints fall inside the image."""
    if not (z > 0).all():
        return False
    inb = ((j2d[:, 0] >= 0) & (j2d[:, 0] < w) & (j2d[:, 1] >= 0) & (j2d[:, 1] < h))
    return inb.mean() >= 0.5


def convert_frame(fid, anno, calib, mano_models, world_to_v0, device):
    rm = anno["raw_mano"][fid]
    hands = []
    for hand_id, (side, pref) in enumerate([("right", "rh"), ("left", "lh")]):
        pose = rm[f"{pref}__pose_coeffs"][0].numpy()
        betas = rm[f"{pref}__betas"][0].numpy()
        tsl = rm[f"{pref}__tsl"][0].numpy()
        j_world_raw, global_orient = mano_world_joints(
            mano_models[side], pose, betas, tsl, device)
        # rebase original-world joints + global_orient into view0(cam0) world frame
        j_h = np.concatenate([j_world_raw, np.ones((21, 1), np.float32)], 1)
        j_world = (world_to_v0 @ j_h.T).T[:, :3].astype(np.float32)
        Rgo = world_to_v0[:3, :3] @ R.from_rotvec(global_orient).as_matrix()
        global_orient = R.from_matrix(Rgo).as_rotvec().astype(np.float32)

        per_view = {}
        for k in range(len(VIEW_ORDER)):
            Rwc = calib["R_world_to_cam"][k]; twc = calib["t_world_to_cam"][k]
            pc = (Rwc @ j_world.T).T + twc
            j2d = project(calib["K"][k], pc)
            w, h = calib["img_size"][k]
            if not view_sees_hand(j2d, pc[:, 2], w, h):
                continue
            per_view[k] = {
                "bbox": bbox_from_2d(j2d, w, h),
                "joints_2d": j2d.astype(np.float32),
                "joints_3d": pc.astype(np.float32),
            }

        # ego view (dynamic): use THIS frame's ego extrinsic, rebased into the view0 world frame.
        E0_inv = np.linalg.inv(world_to_v0)  # world_to_v0 == E_0 (cam0 original world->cam)
        w2c_ego = np.asarray(anno["cam_extr"][EGO_NAME][fid], dtype=np.float64) @ E0_inv
        Rwc_e, twc_e = w2c_ego[:3, :3], w2c_ego[:3, 3]
        pc_e = (Rwc_e @ j_world.T).T + twc_e
        j2d_e = project(calib["K"][EGO_VIEW], pc_e)
        we, he = calib["img_size"][EGO_VIEW]
        if view_sees_hand(j2d_e, pc_e[:, 2], we, he):
            per_view[EGO_VIEW] = {
                "bbox": bbox_from_2d(j2d_e, we, he),
                "joints_2d": j2d_e.astype(np.float32),
                "joints_3d": pc_e.astype(np.float32),
                "extrinsics": np.concatenate(
                    [Rwc_e, twc_e[:, None]], axis=1).astype(np.float32),  # (3,4) world->cam
            }
        if not per_view:
            continue  # no view sees this hand
        hands.append({
            "hand_id": hand_id, "side": side,
            "mano": {"global_orient": global_orient,
                     "hand_pose": R.from_quat(pose[1:][:, [1, 2, 3, 0]]).as_rotvec().reshape(-1).astype(np.float32),
                     "betas": betas.astype(np.float32)},
            "views": per_view,
        })
    return {"frame_id": int(fid), "is_annotated": True, "hands": hands}


# ── verification ──────────────────────────────────────────────────────────────

def verify_session(anno, calib, mano_models, world_to_v0, device, n=3):
    checked = 0
    for fid in anno["frame_id_list"]:
        data = convert_frame(fid, anno, calib, mano_models, world_to_v0, device)
        if not data["hands"]:
            continue
        for hand in data["hands"]:
            for k, vw in hand["views"].items():
                j2d = vw["joints_2d"]; w, h = calib["img_size"][k]
                assert (vw["joints_3d"][:, 2] > 0).all(), f"z<=0 fid {fid} view {k}"
                inb = ((j2d[:, 0] >= 0) & (j2d[:, 0] < w) &
                       (j2d[:, 1] >= 0) & (j2d[:, 1] < h)).mean()
                assert inb >= 0.5, f"bbox mostly OOB fid {fid} view {k} ({inb:.2f})"
        checked += 1
        if checked >= n:
            break
    return checked


# ── session driver ────────────────────────────────────────────────────────────

def discover_sessions(oakink_root):
    """Return [(name, anno_path, extracted_dir)] for sequences with both pickle
    and an extracted image tree."""
    anno_dir = os.path.join(oakink_root, "anno_preview")
    extr_dir = os.path.join(oakink_root, "data", "extracted")
    out = []
    for p in sorted(glob.glob(os.path.join(anno_dir, "*.pkl"))):
        name = os.path.splitext(os.path.basename(p))[0]
        ed = os.path.join(extr_dir, name)
        if os.path.isdir(ed):
            out.append((name, p, ed))
    return out


def session_is_complete(sess_dir, n_fids):
    if not os.path.exists(os.path.join(sess_dir, "calib.npz")):
        return False
    return len(glob.glob(os.path.join(sess_dir, "frames", "*.npz"))) >= n_fids


def write_image_jpg(src_png, dst_jpg):
    if os.path.exists(dst_jpg):
        return True
    img = cv2.imread(src_png)
    if img is None:
        return False
    os.makedirs(os.path.dirname(dst_jpg), exist_ok=True)
    cv2.imwrite(dst_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def convert_session(name, anno_path, extr_dir, anno, out_dir, mano_models,
                    device, verify, overwrite, show_progress=True, img_threads=8):
    fids = anno["frame_id_list"]
    sess_dir = os.path.join(out_dir, name)
    if not overwrite and session_is_complete(sess_dir, len(fids)):
        return len(fids), None, 0  # n_hands=None signals skipped

    calib, world_to_v0 = build_calib(anno)
    os.makedirs(os.path.join(sess_dir, "frames"), exist_ok=True)
    np.savez(os.path.join(sess_dir, "calib.npz"), **calib)
    if verify:
        verify_session(anno, calib, mano_models, world_to_v0, device)

    name_to_camid = {v: k for k, v in anno["cam_def"].items()}
    cam_ids = [name_to_camid[v] for v in VIEW_ORDER] + [name_to_camid[EGO_NAME]]  # view0..view3

    # ---- 1) images: write them FIRST, in parallel ----------------------------------------
    # The image src->dst mapping depends only on (fid, view), not on convert_frame, so it can be
    # built up front and parallelised. cv2 imread/encode/imwrite release the GIL, so a thread
    # pool gives near-linear speedup on this I/O-bound step. Each task is an independent
    # PNG->JPG conversion, so the produced files are identical to writing them serially.
    # Images are written BEFORE the frame npz so that the npz count (session_is_complete's
    # completion marker) only reaches n_fids once the images are already done -> resume stays
    # correct (an interrupted run re-runs and write_image_jpg skips already-converted images).
    img_tasks = []  # (src_png, dst_jpg)
    for fid in fids:
        out_idx = f"{int(fid):06d}"
        for k, cam_id in enumerate(cam_ids):
            src = os.path.join(extr_dir, cam_id, f"{int(fid):06d}.png")
            if os.path.exists(src):
                img_tasks.append(
                    (src, os.path.join(sess_dir, "images", f"view{k}", f"{out_idx}.jpg")))

    if img_tasks:
        if img_threads and img_threads > 1:
            with ThreadPoolExecutor(max_workers=img_threads) as ex:
                results = ex.map(lambda t: write_image_jpg(*t), img_tasks)
                for _ in tqdm(results, total=len(img_tasks), desc=f"{name[:30]} imgs",
                              leave=False, disable=not show_progress):
                    pass
        else:
            for t in tqdm(img_tasks, desc=f"{name[:30]} imgs",
                          leave=False, disable=not show_progress):
                write_image_jpg(*t)

    # ---- 2) frame npz: written LAST (= completion marker) ---------------------------------
    n_hands = 0
    for fid in tqdm(fids, desc=name[:40], leave=False, disable=not show_progress):
        data = convert_frame(fid, anno, calib, mano_models, world_to_v0, device)
        n_hands += len(data["hands"])
        out_idx = f"{int(fid):06d}"
        np.savez(os.path.join(sess_dir, "frames", f"{out_idx}.npz"),
                 data=np.array(data, dtype=object))
    return len(fids), n_hands, len(fids)


# ── parallel worker ───────────────────────────────────────────────────────────

# Per-process state, built once on first task (spawned workers each get their own).
_WORKER = {}


def _worker_init(mano_parent, verify, overwrite, out_root, tqdm_lock, img_threads):
    # Share the parent's tqdm lock so any tqdm.write() from a worker is serialized
    # against the parent's live progress bar (spawned procs don't inherit it).
    tqdm.set_lock(tqdm_lock)
    # Silence noisy startup output (smplx "10 shape coefficients", pynvml
    # FutureWarning) so it can't smear the parent's bar.
    import warnings
    warnings.filterwarnings("ignore")
    # CPU-only inside workers: many processes sharing one GPU would contend and
    # CUDA + spawn is wasteful. MANO on CPU per-process is cheap.
    torch.set_num_threads(1)  # avoid oversubscription when N procs each use BLAS
    _WORKER["device"] = torch.device("cpu")
    _WORKER["mano"] = build_mano_models(mano_parent, _WORKER["device"])
    _WORKER["verify"] = verify
    _WORKER["overwrite"] = overwrite
    _WORKER["out_root"] = out_root
    _WORKER["img_threads"] = img_threads


def _worker_convert(session):
    name, anno_path, extr_dir = session
    try:
        anno = pickle.load(open(anno_path, "rb"))
        nf, nh, _ = convert_session(
            name, anno_path, extr_dir, anno, _WORKER["out_root"], _WORKER["mano"],
            _WORKER["device"], _WORKER["verify"], _WORKER["overwrite"],
            show_progress=False,  # only the pool's outer session bar in workers
            img_threads=_WORKER["img_threads"])
        return name, nf, nh, None
    except Exception as e:  # keep the pool alive; report and continue
        return name, 0, 0, f"{type(e).__name__}: {e}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oakink_root", default=DEFAULT_OAKINK_ROOT)
    ap.add_argument("--out_root", default=None,
                    help="default: <oakink_root>/oakink_sessions")
    ap.add_argument("--mano_pkl_dir", default=DEFAULT_MANO_PKL_DIR,
                    help="dir holding MANO_{LEFT,RIGHT}.pkl (or a .../mano dir)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="MANO device for serial (--workers 1) runs; workers are CPU-only")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes over sessions (each converts whole "
                         "sessions independently). >1 forces CPU MANO.")
    ap.add_argument("--img_threads", type=int, default=8,
                    help="threads per session for the PNG->JPG image conversion (cv2 releases "
                         "the GIL, so this speeds up the I/O-bound image step). 1 = serial.")
    ap.add_argument("--limit", type=int, default=None, help="first N sessions (smoke test)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-convert complete sessions (default: skip/resume)")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(args.oakink_root, "sessions")
    os.makedirs(out_root, exist_ok=True)
    mano_parent = setup_mano_dir(args.mano_pkl_dir)

    sessions = discover_sessions(args.oakink_root)
    if args.limit:
        sessions = sessions[: args.limit]
    print(f"Converting {len(sessions)} sessions -> {out_root} "
          f"({args.workers} worker{'s' if args.workers != 1 else ''})")

    grand = {"sessions": 0, "skipped": 0, "frames": 0, "hands": 0, "failed": 0}

    def tally(name, nf, nh, err):
        if err is not None:
            tqdm.write(f"  {name}: FAILED — {err}")
            grand["failed"] += 1
        elif nh is None:
            tqdm.write(f"  {name}: {nf} frames — SKIPPED (already converted)")
            grand["skipped"] += 1
        else:
            tqdm.write(f"  {name}: {nf} frames, {nh} hand-instances")
            grand["hands"] += nh
        grand["sessions"] += 1; grand["frames"] += nf

    if args.workers <= 1:
        device = torch.device(args.device)
        mano_models = build_mano_models(mano_parent, device)
        for session in tqdm(sessions, desc="sessions"):
            name, anno_path, extr_dir = session
            anno = pickle.load(open(anno_path, "rb"))
            nf, nh, _ = convert_session(
                name, anno_path, extr_dir, anno, out_root, mano_models,
                device, args.verify, args.overwrite, img_threads=args.img_threads)
            tally(name, nf, nh, None)
    else:
        # process pool over sessions; each worker builds MANO once (CPU).
        ctx = mp.get_context("spawn")  # safe with torch; avoids fork+CUDA hazards
        # A Manager RLock is picklable and shared across processes; both the
        # parent bar and any worker tqdm.write() acquire it, so terminal output
        # never interleaves/corrupts the bar.
        manager = ctx.Manager()
        tqdm_lock = manager.RLock()
        tqdm.set_lock(tqdm_lock)
        with ctx.Pool(
            processes=args.workers, initializer=_worker_init,
            initargs=(mano_parent, args.verify, args.overwrite, out_root, tqdm_lock,
                      args.img_threads),
        ) as pool:
            # imap_unordered: results stream back as sessions finish (good for tqdm)
            for name, nf, nh, err in tqdm(
                pool.imap_unordered(_worker_convert, sessions),
                total=len(sessions), desc="sessions", lock_args=(True,),
            ):
                tally(name, nf, nh, err)

    print(f"\nDone. {grand['sessions']} sessions ({grand['skipped']} skipped, "
          f"{grand['failed']} failed), {grand['frames']} frames, "
          f"{grand['hands']} hand-instances -> {out_root}")


if __name__ == "__main__":
    main()

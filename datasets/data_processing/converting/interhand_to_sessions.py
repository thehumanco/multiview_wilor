#!/usr/bin/env python3
"""
Convert InterHand2.6M 5fps batch1 -> the project's multi-view "session" format.

InterHand2.6M ships a static multi-camera capture rig, COCO-style image entries,
per-capture camera calibration, per-frame world-frame 3D joints, and NeuralAnnot
MANO parameters. This converter reconstructs one true multi-view session per
(capture, sequence):

    <out>/<split>/Capture<X>__<seq_name>/
        calib.npz   { K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3),
                      t_world_to_cam:(V,3), img_size:(V,2) }
        frames/NNNNNN.npz
        images/view0/NNNNNN.jpg ... viewK/NNNNNN.jpg

Geometry assumptions, matching the canonical InterHand/NeuralAnnot layout:
  * camrot is world->cam rotation and campos is camera position in world (mm).
    Raw world->cam translation is t = -camrot @ campos, converted to meters.
  * world_coord is 42 joints in millimeters, right hand first then left hand.
    InterHand native per-hand order is remapped to the HO-3D/project session
    order: wrist, thumb base->tip, index base->tip, middle base->tip,
    ring base->tip, pinky base->tip.
  * MANO pose is 48 axis-angle values; pose[:3] is world-frame global_orient.
    pose[3:48] stays in native MANO 15-joint pose-vector order, matching
    HO-3D handPose[3:] and OakInk raw_mano pose_coeffs[1:].
  * Output sessions use meters and are rebased so view0 is the world frame.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

DEFAULT_INTERHAND_ROOT = "/lambda/nfs/hfm/qasim/hand_kp_dataset/interhands2.6m"
IMAGE_SUBDIR = "InterHand2.6M_5fps_batch1/images"
SPLITS = ("train", "val", "test")
SIDES = (("right", 0, slice(0, 21)), ("left", 1, slice(21, 42)))

# InterHand skeleton.txt per-hand order is tip->base for thumb/index/middle/ring/
# pinky, then wrist. HO-3D handJoints3D order is MANO's 16 joints plus fingertips:
# wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3, then
# thumb/index/middle/ring/pinky tips. Apply this before projecting so joints_2d
# and joints_3d share the same keypoint layout as HO-3D.
INTERHAND_TO_HO3D = np.array([
    20,
    7, 6, 5,
    11, 10, 9,
    19, 18, 17,
    15, 14, 13,
    3, 2, 1,
    0, 4, 8, 12, 16,
], dtype=np.int64)

# This names the 21-joint arrays after INTERHAND_TO_HO3D is applied. It is the
# same keypoint order emitted by HO-3D handJoints3D.
SESSION_JOINT_NAMES = (
    "wrist",
    "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
    "thumb4", "index4", "middle4", "ring4", "pinky4",
)

# 45-D mano.hand_pose is not stored in SESSION_JOINT_NAMES order. HO-3D keeps
# native MANO handPose[3:] order, so InterHand and OakInk must keep it too.
MANO_HAND_POSE_JOINT_NAMES = (
    "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
)


# -- rotation helpers ---------------------------------------------------------

try:
    from scipy.spatial.transform import Rotation as _SciRotation
except Exception:  # pragma: no cover - exercised only in minimal envs
    _SciRotation = None


def rotvec_to_matrix(rv):
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    if _SciRotation is not None:
        return _SciRotation.from_rotvec(rv).as_matrix()
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rv / theta
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def matrix_to_rotvec(mat):
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    if _SciRotation is not None:
        return _SciRotation.from_matrix(mat).as_rotvec()
    trace = np.trace(mat)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(float(cos_theta))
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    if abs(math.pi - theta) < 1e-5:
        vals = np.maximum((np.diag(mat) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(vals)
        if mat[0, 1] < 0:
            axis[1] = -axis[1]
        if mat[0, 2] < 0:
            axis[2] = -axis[2]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= norm
        return axis * theta
    skew = np.array([
        mat[2, 1] - mat[1, 2],
        mat[0, 2] - mat[2, 0],
        mat[1, 0] - mat[0, 1],
    ])
    return skew * (theta / (2.0 * math.sin(theta)))


# -- split loading / indexing -------------------------------------------------

def _json_path(interhand_root, split, suffix):
    return os.path.join(
        interhand_root, "annotations", split,
        f"InterHand2.6M_{split}_{suffix}.json",
    )


def load_json_checked(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not os.access(path, os.R_OK):
        raise PermissionError(
            f"{path} is not readable. Expected qasim to run: "
            f"chmod -R o+r {os.path.dirname(os.path.dirname(path))}"
        )
    with open(path, "r") as f:
        return json.load(f)


def load_split(interhand_root, split):
    """Load and compact one split's annotations.

    Returns a dict with the original camera/joint/MANO maps plus compact session
    entries derived from data.json images and annotations.
    """
    data = load_json_checked(_json_path(interhand_root, split, "data"))
    camera = load_json_checked(_json_path(interhand_root, split, "camera"))
    joint_3d = load_json_checked(_json_path(interhand_root, split, "joint_3d"))
    mano = load_json_checked(_json_path(interhand_root, split, "MANO_NeuralAnnot"))

    annotations = {}
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id is not None and image_id not in annotations:
            annotations[image_id] = ann

    sessions = defaultdict(lambda: {"images": [], "frames": set(), "cameras": set()})
    image_root = os.path.join(interhand_root, IMAGE_SUBDIR, split)
    for img in data.get("images", []):
        capture = img.get("capture")
        seq_name = img.get("seq_name")
        cam_id = img.get("camera")
        frame_idx = img.get("frame_idx")
        file_name = img.get("file_name")
        if capture is None or seq_name is None or cam_id is None or frame_idx is None or not file_name:
            continue
        src = os.path.join(image_root, file_name)
        key = (int(capture), str(seq_name))
        entry = {
            "capture": int(capture),
            "seq_name": str(seq_name),
            "camera": int(cam_id),
            "frame_idx": int(frame_idx),
            "file_name": file_name,
            "path": src,
            "width": int(img.get("width", 0) or 0),
            "height": int(img.get("height", 0) or 0),
            "ann": annotations.get(img.get("id"), {}),
        }
        sessions[key]["images"].append(entry)
        sessions[key]["frames"].add(entry["frame_idx"])
        sessions[key]["cameras"].add(entry["camera"])

    del data
    out_sessions = []
    for (capture, seq_name), info in sessions.items():
        out_sessions.append({
            "split": split,
            "capture": capture,
            "seq_name": seq_name,
            "name": f"Capture{capture}__{seq_name}",
            "images": info["images"],
            "frames": sorted(info["frames"]),
            "cameras": sorted(info["cameras"]),
        })
    out_sessions.sort(key=lambda x: (x["capture"], x["seq_name"]))
    return {"split": split, "camera": camera, "joint_3d": joint_3d, "mano": mano,
            "sessions": out_sessions}


def split_names(which):
    return list(SPLITS) if which == "all" else [which]


# -- geometry / frame conversion ---------------------------------------------

def _capture_map(mapping, capture):
    return mapping.get(str(capture), mapping.get(capture, {}))


def _frame_map(mapping, capture, fid):
    return _capture_map(mapping, capture).get(str(fid), _capture_map(mapping, capture).get(fid))


def camera_value(cam_block, field, cam_id):
    block = cam_block[field]
    return block.get(str(cam_id), block.get(cam_id))


def make_w2c(Rwc, twc):
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = Rwc
    E[:3, 3] = twc
    return E


def build_calib(split_data, session):
    cam_block = _capture_map(split_data["camera"], session["capture"])
    view_ids = session["cameras"]
    V = len(view_ids)
    K = np.zeros((V, 3, 3), dtype=np.float64)
    dist = np.zeros((V, 5), dtype=np.float64)
    Rwc = np.zeros((V, 3, 3), dtype=np.float64)
    twc = np.zeros((V, 3), dtype=np.float64)
    img_size = np.zeros((V, 2), dtype=np.int64)

    entries = session["images"]
    size_by_cam = {}
    for entry in entries:
        if entry["width"] > 0 and entry["height"] > 0:
            size_by_cam.setdefault(entry["camera"], (entry["width"], entry["height"]))

    raw_E = []
    for cam_id in view_ids:
        R_raw = np.asarray(camera_value(cam_block, "camrot", cam_id), dtype=np.float64).reshape(3, 3)
        C_raw = np.asarray(camera_value(cam_block, "campos", cam_id), dtype=np.float64).reshape(3)
        t_raw = (-R_raw @ C_raw) * 1e-3
        raw_E.append(make_w2c(R_raw, t_raw))

    E0_inv = np.linalg.inv(raw_E[0])
    for k, cam_id in enumerate(view_ids):
        focal = np.asarray(camera_value(cam_block, "focal", cam_id), dtype=np.float64).reshape(2)
        princpt = np.asarray(camera_value(cam_block, "princpt", cam_id), dtype=np.float64).reshape(2)
        K[k] = np.array([[focal[0], 0.0, princpt[0]], [0.0, focal[1], princpt[1]], [0.0, 0.0, 1.0]])
        w, h = size_by_cam.get(cam_id, (512, 334))
        img_size[k] = (w, h)
        E_rebased = raw_E[k] @ E0_inv
        Rwc[k] = E_rebased[:3, :3]
        twc[k] = E_rebased[:3, 3]

    calib = {"K": K, "dist": dist, "R_world_to_cam": Rwc,
             "t_world_to_cam": twc, "img_size": img_size}
    return calib, raw_E[0]


def project(K, pts_cam):
    uv = (K @ pts_cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def bbox_from_2d(j2d, w, h, margin=0.15):
    x0, y0 = j2d.min(0); x1, y1 = j2d.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0 -= margin * bw; x1 += margin * bw; y0 -= margin * bh; y1 += margin * bh
    return np.array([max(0, x0), max(0, y0), min(w, x1), min(h, y1)], dtype=np.float32)


def view_sees_hand(j2d, z, w, h):
    if not np.isfinite(j2d).all() or not np.isfinite(z).all():
        return False
    if not (z > 0).all():
        return False
    inb = ((j2d[:, 0] >= 0) & (j2d[:, 0] < w) & (j2d[:, 1] >= 0) & (j2d[:, 1] < h))
    return inb.mean() >= 0.5


def frame_is_hand_type(joint_info, side):
    hand_type = joint_info.get("hand_type")
    if hand_type in (None, "interacting"):
        return True
    return hand_type == side


def valid_side(joint_info, side, side_slice):
    if not frame_is_hand_type(joint_info, side):
        return False
    valid = np.asarray(joint_info.get("joint_valid", []), dtype=np.float32)
    if valid.size >= 42:
        return bool((valid[side_slice] > 0).all())
    return True


def entry_valid_side(entry, side, side_slice):
    ann = entry.get("ann") or {}
    if not frame_is_hand_type(ann, side):
        return False
    if ann.get("hand_type_valid", 1) in (0, False):
        return False
    valid = np.asarray(ann.get("joint_valid", []), dtype=np.float32)
    if valid.size >= 42:
        return bool((valid[side_slice] > 0).all())
    return True


def mano_for_side(mano_frame, side, world_to_v0):
    zeros = {
        "global_orient": np.zeros(3, dtype=np.float32),
        "hand_pose": np.zeros(45, dtype=np.float32),
        "betas": np.zeros(10, dtype=np.float32),
    }
    if not isinstance(mano_frame, dict) or mano_frame.get(side) is None:
        return zeros, False
    raw = mano_frame[side]
    pose = np.asarray(raw.get("pose", []), dtype=np.float64).reshape(-1)
    shape = np.asarray(raw.get("shape", []), dtype=np.float64).reshape(-1)
    if pose.size < 48 or shape.size < 10:
        return zeros, False
    Rgo = world_to_v0[:3, :3] @ rotvec_to_matrix(pose[:3])
    return {
        "global_orient": matrix_to_rotvec(Rgo).astype(np.float32),
        "hand_pose": pose[3:48].astype(np.float32),
        "betas": shape[:10].astype(np.float32),
    }, True


def convert_frame(fid, split_data, session, calib, world_to_v0, images_by_frame_cam):
    joint_info = _frame_map(split_data["joint_3d"], session["capture"], fid)
    mano_frame = _frame_map(split_data["mano"], session["capture"], fid)
    hands = []
    any_mano = False
    if not joint_info:
        return {"frame_id": int(fid), "is_annotated": False, "hands": hands}

    world_coord = np.asarray(joint_info.get("world_coord", []), dtype=np.float64)
    if world_coord.size < 42 * 3:
        return {"frame_id": int(fid), "is_annotated": False, "hands": hands}
    world_coord = world_coord.reshape(-1, 3)[:42] * 1e-3

    for side, hand_id, side_slice in SIDES:
        if not valid_side(joint_info, side, side_slice):
            continue
        jw_raw = world_coord[side_slice][INTERHAND_TO_HO3D]
        jw_h = np.concatenate([jw_raw, np.ones((21, 1), dtype=np.float64)], axis=1)
        jw = (world_to_v0 @ jw_h.T).T[:, :3]
        mano, has_mano = mano_for_side(mano_frame, side, world_to_v0)
        any_mano = any_mano or has_mano

        per_view = {}
        for k, cam_id in enumerate(session["cameras"]):
            entry = images_by_frame_cam.get((fid, cam_id))
            if entry is None:
                continue
            if not entry_valid_side(entry, side, side_slice):
                continue
            Rwc = calib["R_world_to_cam"][k]; twc = calib["t_world_to_cam"][k]
            pc = (Rwc @ jw.T).T + twc
            j2d = project(calib["K"][k], pc)
            w, h = calib["img_size"][k]
            if not view_sees_hand(j2d, pc[:, 2], w, h):
                continue
            per_view[k] = {
                "bbox": bbox_from_2d(j2d, w, h),
                "joints_2d": j2d.astype(np.float32),
                "joints_3d": pc.astype(np.float32),
            }
        if per_view:
            hands.append({
                "hand_id": hand_id,
                "side": side,
                "mano": mano,
                "views": per_view,
            })
    return {"frame_id": int(fid), "is_annotated": bool(any_mano), "hands": hands}


# -- verification / IO --------------------------------------------------------

def verify_frame(data, calib, atol_world=1e-3, atol_px=1.0):
    for hand in data["hands"]:
        worlds = []
        for k, view in hand["views"].items():
            j3d = view["joints_3d"].astype(np.float64)
            reproj = project(calib["K"][k], j3d)
            err_px = float(np.nanmax(np.linalg.norm(reproj - view["joints_2d"], axis=1)))
            assert err_px < atol_px, f"reprojection error {err_px:.3f}px frame {data['frame_id']} view {k}"
            Rwc, twc = calib["R_world_to_cam"][k], calib["t_world_to_cam"][k]
            worlds.append((Rwc.T @ (j3d - twc).T).T)
        if len(worlds) >= 2:
            base = worlds[0]
            spread = max(float(np.nanmax(np.linalg.norm(w - base, axis=1))) for w in worlds[1:])
            assert spread < atol_world, f"world spread {spread*1000:.3f}mm frame {data['frame_id']}"


def verify_session(split_data, session, calib, world_to_v0, images_by_frame_cam, n=3):
    checked = 0
    for fid in session["frames"]:
        data = convert_frame(fid, split_data, session, calib, world_to_v0, images_by_frame_cam)
        if not data["hands"]:
            continue
        verify_frame(data, calib)
        checked += 1
        if checked >= n:
            break
    return checked


def session_is_complete(sess_dir, n_fids):
    if not os.path.exists(os.path.join(sess_dir, "calib.npz")):
        return False
    return len(glob.glob(os.path.join(sess_dir, "frames", "*.npz"))) >= n_fids


def write_image(src, dst, symlink):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def image_lookup(session):
    lookup = {}
    for entry in session["images"]:
        lookup.setdefault((entry["frame_idx"], entry["camera"]), entry)
    return lookup


def convert_session(split_data, session, out_split_dir, symlink, verify, overwrite,
                    progress=None, show_progress=True):
    sess_dir = os.path.join(out_split_dir, session["name"])
    if not overwrite and session_is_complete(sess_dir, len(session["frames"])):
        if progress:
            progress.add_session(len(session["frames"]))
        return session["name"], len(session["cameras"]), len(session["frames"]), None, 0

    calib, world_to_v0 = build_calib(split_data, session)
    images_by_frame_cam = image_lookup(session)
    os.makedirs(os.path.join(sess_dir, "frames"), exist_ok=True)
    np.savez(os.path.join(sess_dir, "calib.npz"), **calib)

    if verify:
        verify_session(split_data, session, calib, world_to_v0, images_by_frame_cam)

    n_hands = 0
    n_annotated = 0
    for fid in tqdm(session["frames"], desc=session["name"][:40], leave=False, disable=not show_progress):
        data = convert_frame(fid, split_data, session, calib, world_to_v0, images_by_frame_cam)
        n_hands += len(data["hands"])
        n_annotated += int(data["is_annotated"])
        out_idx = f"{int(fid):06d}"
        np.savez(os.path.join(sess_dir, "frames", f"{out_idx}.npz"),
                 data=np.array(data, dtype=object))
        for k, cam_id in enumerate(session["cameras"]):
            entry = images_by_frame_cam.get((fid, cam_id))
            if entry is not None:
                if os.path.exists(entry["path"]):
                    write_image(entry["path"], os.path.join(
                        sess_dir, "images", f"view{k}", f"{out_idx}.jpg"), symlink)
        if progress:
            progress.add_frame()
    if progress:
        progress.add_session(0)
    return session["name"], len(session["cameras"]), len(session["frames"]), n_annotated, n_hands


# -- heartbeat ----------------------------------------------------------------

class Progress:
    def __init__(self, total_sessions, total_frames, interval=30.0,
                 shared_sessions=None, shared_frames=None):
        self.total_sessions = int(total_sessions)
        self.total_frames = int(total_frames)
        self.interval = float(interval)
        self.shared_sessions = shared_sessions
        self.shared_frames = shared_frames
        self.sessions = 0
        self.frames = 0
        self.lock = threading.Lock()
        self.start = time.time()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def add_frame(self, n=1):
        if self.shared_frames is not None:
            with self.shared_frames.get_lock():
                self.shared_frames.value += n
            return
        with self.lock:
            self.frames += n

    def add_session(self, skipped_frames=0):
        if self.shared_sessions is not None and self.shared_frames is not None:
            with self.shared_sessions.get_lock():
                self.shared_sessions.value += 1
            if skipped_frames:
                with self.shared_frames.get_lock():
                    self.shared_frames.value += skipped_frames
            return
        with self.lock:
            self.sessions += 1
            self.frames += skipped_frames

    def snapshot(self):
        if self.shared_sessions is not None and self.shared_frames is not None:
            with self.shared_sessions.get_lock():
                sessions = self.shared_sessions.value
            with self.shared_frames.get_lock():
                frames = self.shared_frames.value
            return sessions, frames, time.time() - self.start
        with self.lock:
            return self.sessions, self.frames, time.time() - self.start

    def _run(self):
        while not self.stop_event.wait(self.interval):
            sessions, frames, elapsed = self.snapshot()
            fps = frames / elapsed if elapsed > 0 else 0.0
            remaining = max(0, self.total_frames - frames)
            eta = remaining / fps if fps > 0 else float("inf")
            eta_txt = "unknown" if not math.isfinite(eta) else time.strftime("%H:%M:%S", time.gmtime(eta))
            elapsed_txt = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            tqdm.write(
                f"[heartbeat] sessions {sessions}/{self.total_sessions}, "
                f"frames {frames}/{self.total_frames}, {fps:.2f} frames/sec, "
                f"ETA {eta_txt}, elapsed {elapsed_txt}",
                file=sys.stderr,
            )


# -- worker pool --------------------------------------------------------------

_WORKER = {}


def _worker_init_loaded(split_data, out_split_dir, symlink, verify, overwrite):
    _WORKER["split_data"] = split_data
    _WORKER["out_split_dir"] = out_split_dir
    _WORKER["symlink"] = symlink
    _WORKER["verify"] = verify
    _WORKER["overwrite"] = overwrite
    _WORKER["progress"] = None


def _worker_init_loaded_with_progress(split_data, out_split_dir, symlink, verify,
                                      overwrite, shared_sessions, shared_frames):
    _worker_init_loaded(split_data, out_split_dir, symlink, verify, overwrite)
    _WORKER["progress"] = Progress(
        total_sessions=0, total_frames=0,
        shared_sessions=shared_sessions, shared_frames=shared_frames,
    )


def _worker_convert(session):
    try:
        return (*convert_session(
            _WORKER["split_data"], session, _WORKER["out_split_dir"],
            _WORKER["symlink"], _WORKER["verify"], _WORKER["overwrite"],
            progress=_WORKER["progress"], show_progress=False,
        ), None)
    except Exception as e:
        return session["name"], 0, 0, 0, 0, f"{type(e).__name__}: {e}"


# -- main ---------------------------------------------------------------------

def run_split(args, split, out_root, grand):
    out_split_dir = os.path.join(out_root, split)
    os.makedirs(out_split_dir, exist_ok=True)

    print(f"\n[{split}] Loading InterHand annotations...", flush=True)
    split_data = load_split(args.interhand_root, split)
    sessions = split_data["sessions"]
    if args.limit:
        sessions = sessions[:args.limit]
    total_frames = sum(len(s["frames"]) for s in sessions)
    print(f"\n[{split}] Converting {len(sessions)} sessions, {total_frames} frames -> {out_split_dir}")

    if args.workers <= 1:
        with Progress(len(sessions), total_frames) as progress:
            for session in tqdm(sessions, desc=f"{split} sessions"):
                try:
                    name, V, nf, nann, nh = convert_session(
                        split_data, session, out_split_dir, args.symlink_images,
                        args.verify, args.overwrite, progress=progress)
                    err = None
                except Exception as e:
                    name, V, nf, nann, nh, err = session["name"], 0, 0, 0, 0, f"{type(e).__name__}: {e}"
                tally(split, name, V, nf, nann, nh, err, grand)
    else:
        # No torch/CUDA here, so prefer fork on Linux: the large split JSONs
        # stay shared copy-on-write instead of being reloaded in every worker.
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)
        shared_sessions = ctx.Value("i", 0)
        shared_frames = ctx.Value("i", 0)
        work_sessions = sorted(sessions, key=lambda s: len(s["frames"]), reverse=True)
        with Progress(
            len(sessions), total_frames,
            shared_sessions=shared_sessions, shared_frames=shared_frames,
        ) as progress:
            with ctx.Pool(
                processes=args.workers,
                initializer=_worker_init_loaded_with_progress,
                initargs=(split_data, out_split_dir, args.symlink_images,
                          args.verify, args.overwrite, shared_sessions, shared_frames),
            ) as pool:
                for name, V, nf, nann, nh, err in tqdm(
                    pool.imap_unordered(_worker_convert, work_sessions),
                    total=len(sessions), desc=f"{split} sessions",
                ):
                    tally(split, name, V, nf, nann, nh, err, grand)


def tally(split, name, V, nf, nann, nh, err, grand):
    if err:
        tqdm.write(f"  [{split}] {name}: FAILED - {err}")
        grand["failed"] += 1
    elif nann is None:
        tqdm.write(f"  [{split}] {name}: {V} views, {nf} frames - SKIPPED")
        grand["skipped"] += 1
    else:
        tqdm.write(f"  [{split}] {name}: {V} views, {nann}/{nf} annotated frames, {nh} hand-instances")
        grand["annotated"] += nann
        grand["hands"] += nh
    grand["sessions"] += 1
    grand["frames"] += nf
    grand["views"] += V


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interhand_root", default=DEFAULT_INTERHAND_ROOT)
    ap.add_argument("--out_root", default=None,
                    help="default: <interhand_root>/interhand_sessions")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    img = ap.add_mutually_exclusive_group()
    img.add_argument("--symlink-images", dest="symlink_images", action="store_true",
                     help="symlink JPGs into the session tree (default)")
    img.add_argument("--copy-images", dest="symlink_images", action="store_false",
                     help="copy JPGs into the session tree")
    ap.set_defaults(symlink_images=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes over sessions; each worker loads the split JSONs")
    ap.add_argument("--limit", type=int, default=None, help="first N sessions per split")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-convert complete sessions (default: skip/resume)")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(args.interhand_root, "interhand_sessions")
    os.makedirs(out_root, exist_ok=True)

    grand = {"sessions": 0, "skipped": 0, "failed": 0, "frames": 0,
             "annotated": 0, "hands": 0, "views": 0}
    for split in split_names(args.split):
        run_split(args, split, out_root, grand)

    print(f"\nDone. {grand['sessions']} sessions ({grand['skipped']} skipped, "
          f"{grand['failed']} failed), {grand['frames']} frames "
          f"({grand['annotated']} annotated), {grand['hands']} hand-instances, "
          f"{grand['views']} total views -> {out_root}")


if __name__ == "__main__":
    main()

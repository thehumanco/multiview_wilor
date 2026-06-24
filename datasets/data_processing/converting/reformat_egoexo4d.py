#!/usr/bin/env python3
"""Reformat EgoExo4D hand ``ego_pose`` annotations into the project's multi-view "session"
format (see ``reformat_arctic.py`` / ``oakink_to_sessions.py`` for sibling converters and
``session_dataset.py`` for the consumer).

EgoExo4D records each take with one head-mounted Aria (ego) camera + several static GoPro
(exo) cameras. The ``ego_pose/<split>/hand/annotation`` files are the MANUALLY-annotated
keyframes: per camera they hold 2D hand keypoints (each tagged ``placement: manual|auto``) and
a triangulated ``annotation3D`` of 21 joints per hand in the capture WORLD frame. There is NO
MANO fit, so the emitted hands are keypoint-only (``has_mano=False``) — the dataloader masks the
MANO-parameter losses and supervises purely on the 2D/3D keypoints.

Per the project guidance we keep only views where a human actually labelled the hand: a (hand,
view) pair is emitted only if that camera has >= ``--min-view-manual`` manually-placed 2D
keypoints for the hand. Joints that failed triangulation (absent from ``annotation3D``) are
written with confidence 0 so they are not supervised; the wrist (joint 0) is required because
the 3D loss is root-relative.

Cameras (verified against the data):
  * Exo (GoPro) frames on disk are already PINHOLE/undistorted — pinhole projection of the 3D
    matches the stored 2D to ~1px, and overlays land on the hand. We emit them as STATIC views
    0..N-1 with ``dist=0`` and the take's ``camera_pose`` intrinsics/extrinsics (world->cam).
  * Ego (Aria) RGB is fisheye; we undistort each frame to the 512x512 linear camera (focal 150,
    principal point 255.5) that ``camera_pose`` already provides as the Aria intrinsics, using
    ``projectaria_tools`` + the take VRS (identical pipeline to the yolo26 reformatter). The Aria
    head moves every frame, so it is the DYNAMIC last view (index N): its per-frame world->cam
    is written into each frame npz under ``views[N]["extrinsics"]`` and flagged via
    ``dynamic_views``; ``calib.npz`` holds a frame-0 placeholder.

All joints are stored in the smplx-MANO 16-joint kinematic order + 5 fingertips (the "stored"
order ``session_dataset._STORED_TO_OPENPOSE`` expects); joints_3d are in the CAMERA frame.

Output layout (one session == one take):
    <out>/<split>/<take_name>/
        calib.npz   {K:(V,3,3), dist:(V,5), R_world_to_cam:(V,3,3), t_world_to_cam:(V,3),
                     img_size:(V,2), dynamic_views:(1,)=[N]}   # exo views 0..N-1, ego view N
        frames/NNNNNN.npz          # object array "data" (one multi-view frame)
        images/view0/NNNNNN.jpg ... viewN/NNNNNN.jpg

frames/NNNNNN.npz "data" dict:
    {"frame_id": int, "is_annotated": True,
     "hands": [ {"hand_id": 0|1, "side": "right"|"left", "has_mano": False,
                 "views": { k: {"bbox":(4,), "joints_2d":(21,2), "joints_3d":(21,3),
                                "joints_conf":(21,) 1/0 per-joint mask,
                                # ego view only: per-frame world->cam [R|t]
                                ["extrinsics":(3,4)]} } } ] }
"""
import argparse
import json
import multiprocessing as mp
import os
import sys


def _reexec_in_par_env():
    """projectaria_tools only builds into the local pixi env (no aarch64 wheel; see
    projectaria-tools-aarch64-build memory). If we were launched with an interpreter that can't
    import it (e.g. the `prometheus` conda env), transparently re-exec under that pixi python so
    plain `python reformat_egoexo4d.py ...` works. The pixi env also has av/cv2/numpy/tqdm."""
    try:
        import projectaria_tools  # noqa: F401
        return
    except ModuleNotFoundError:
        if os.environ.get("_EGOEXO_REEXEC"):
            raise  # already re-exec'd under the pixi python and it's still missing -> real error
    here = os.path.dirname(os.path.abspath(__file__))
    pixi_py = os.path.normpath(os.path.join(
        here, *[".."] * 4, "projectaria_tools", ".pixi", "envs", "default", "bin", "python"))
    if os.path.exists(pixi_py):
        os.environ["_EGOEXO_REEXEC"] = "1"
        os.execv(pixi_py, [pixi_py, os.path.abspath(__file__), *sys.argv[1:]])
    raise ModuleNotFoundError(
        "projectaria_tools not importable and the pixi env python was not found at "
        f"{pixi_py}. Build it with `pixi run build_python` in src/.../projectaria_tools.")


_reexec_in_par_env()

import av
import cv2
import numpy as np
from tqdm import tqdm

from projectaria_tools.core import calibration, data_provider

DEFAULT_DATA_DIR = "/lambda/nfs/hfm/qasim/hand_kp_dataset/egoexo4d"
DEFAULT_OUT_DIR = "/lambda/nfs/hfm/qasim/hand_kp_dataset/egoexo4d/sessions"
SPLITS = ["train", "val"]

# Aria RGB sensor resolution / linear-undistortion target (matches the camera_pose aria K and
# the yolo26 reformatter: get_linear_camera_calibration(512, 512, 150)).
ARIA_SENSOR = 2880
ARIA_RGB = 1408
ARIA_UNDIST = 512
ARIA_FOCAL = 150

NUM_KP = 21
# stored joint order == smplx MANO 16 kinematic joints + 5 fingertips, expressed as egoexo4d
# joint-name suffixes. (Matches session_dataset._STORED_TO_OPENPOSE.)
STORED_SUFFIXES = [
    "wrist",
    "index_1", "index_2", "index_3",
    "middle_1", "middle_2", "middle_3",
    "pinky_1", "pinky_2", "pinky_3",
    "ring_1", "ring_2", "ring_3",
    "thumb_1", "thumb_2", "thumb_3",
    "thumb_4", "index_4", "middle_4", "ring_4", "pinky_4",
]
# hand_id convention shared with the other converters: 0 == right, 1 == left.
SIDES = [("right", 0), ("left", 1)]


# ── helpers ─────────────────────────────────────────────────────────────────────

def ego_cam_id(take):
    for c in take["capture"]["cameras"]:
        if c["is_ego"] and c["cam_id"].startswith("aria"):
            return c["cam_id"]
    return None


def exo_relpath(fav_entry):
    """frame_aligned_videos exo entries are keyed by stream id (e.g. '0')."""
    for v in fav_entry.values():
        if isinstance(v, dict) and "relative_path" in v:
            return v["relative_path"]
    return None


def project(K, pts_cam):
    """(N,3) camera-frame -> (N,2) pixels (no distortion; exo frames + aria-undistorted are
    pinhole). z<=0 points project to garbage and are masked by the caller via joints_conf."""
    z = np.where(np.abs(pts_cam[:, 2:3]) < 1e-6, 1e-6, pts_cam[:, 2:3])
    uv = (K @ pts_cam.T).T
    return uv[:, :2] / z


def count_manual(cam_ann, names):
    return sum(
        1 for nm in names
        if cam_ann.get(nm) is not None and cam_ann[nm].get("placement") != "auto"
    )


def view_sees_hand(j2d, z, mask, w, h, min_frac=0.3):
    m = (mask > 0) & (z > 0)
    if m.sum() < 3:
        return False
    inb = m & (j2d[:, 0] >= 0) & (j2d[:, 0] < w) & (j2d[:, 1] >= 0) & (j2d[:, 1] < h)
    return inb.sum() >= max(3, min_frac * m.sum())


def bbox_from_joints2d(j2d, conf, w, h, margin=0.2):
    m = conf > 0
    if m.sum() < 2:
        return None
    pts = j2d[m]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0 -= margin * bw; x1 += margin * bw
    y0 -= margin * bh; y1 += margin * bh
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = min(float(w), x1); y1 = min(float(h), y1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# ── geometry pass (no image decode) ──────────────────────────────────────────────

def build_views(take, hand_ann, cam_pose, data_dir, has_ego_media):
    """Return (calib_dict, view_cam_ids, ego_view_idx_or_None, ego_extr_dict)."""
    fav = take["frame_aligned_videos"]
    root = take["root_dir"]
    exo_cams = [
        c for c in sorted(cam_pose.keys())
        if c.startswith("cam") and c in fav and exo_relpath(fav[c]) is not None
        and os.path.exists(os.path.join(data_dir, root, exo_relpath(fav[c])))
    ]
    ego = ego_cam_id(take)
    use_ego = has_ego_media and ego in cam_pose and isinstance(
        cam_pose[ego].get("camera_extrinsics"), dict
    )

    view_cams = list(exo_cams)
    ego_view = None
    ego_extr = None
    V = len(exo_cams) + (1 if use_ego else 0)
    K = np.zeros((V, 3, 3)); dist = np.zeros((V, 5))
    Rwc = np.zeros((V, 3, 3)); twc = np.zeros((V, 3))
    img_size = np.zeros((V, 2), dtype=np.int64)

    for k, cam in enumerate(exo_cams):
        Kk = np.asarray(cam_pose[cam]["camera_intrinsics"], dtype=np.float64)
        ext = np.asarray(cam_pose[cam]["camera_extrinsics"], dtype=np.float64)  # (3,4) world->cam
        K[k] = Kk
        Rwc[k] = ext[:, :3]; twc[k] = ext[:, 3]
        img_size[k] = (int(round(2 * Kk[0, 2])), int(round(2 * Kk[1, 2])))

    if use_ego:
        ego_view = len(exo_cams)
        view_cams.append(ego)
        ego_extr = cam_pose[ego]["camera_extrinsics"]  # {frame_idx(str): (3,4) world->cam}
        K[ego_view] = np.asarray(cam_pose[ego]["camera_intrinsics"], dtype=np.float64)
        f0 = next(iter(ego_extr.values()))
        e0 = np.asarray(f0, dtype=np.float64)
        Rwc[ego_view] = e0[:, :3]; twc[ego_view] = e0[:, 3]
        img_size[ego_view] = (ARIA_UNDIST, ARIA_UNDIST)

    calib = {
        "K": K, "dist": dist, "R_world_to_cam": Rwc, "t_world_to_cam": twc,
        "img_size": img_size,
    }
    if ego_view is not None:
        calib["dynamic_views"] = np.array([ego_view], dtype=np.int64)
    return calib, view_cams, ego_view, ego_extr


def convert_geometry(take, hand_ann, cam_pose, calib, view_cams, ego_view, ego_extr, args):
    """Build per-frame data dicts + the set of (view_idx, frame_id) image cells needed."""
    K = calib["K"]; Rwc = calib["R_world_to_cam"]; twc = calib["t_world_to_cam"]
    img_size = calib["img_size"]
    frames = []
    needed = {k: set() for k in range(len(view_cams))}

    for fkey, flist in hand_ann.items():
        fid = int(fkey)
        fr = flist[0]
        a3 = fr.get("annotation3D", {})
        a2 = fr.get("annotation2D", {})
        hands = []
        for side, hand_id in SIDES:
            names = [f"{side}_{suf}" for suf in STORED_SUFFIXES]
            world = np.zeros((NUM_KP, 3), dtype=np.float64)
            mask = np.zeros(NUM_KP, dtype=np.float32)
            for i, nm in enumerate(names):
                jd = a3.get(nm)
                if jd is not None:
                    world[i] = (jd["x"], jd["y"], jd["z"]); mask[i] = 1.0
            if mask[0] == 0 or mask.sum() < args.min_joints:  # wrist required + min coverage
                continue
            # Joints that failed triangulation have no 3D; snap them to the wrist so they don't
            # project from world-origin into scattered garbage (they carry conf 0 and are masked
            # by every loss, but this keeps the stored 2D/3D + any visualization clean).
            world[mask == 0] = world[0]

            views = {}
            for k, cam in enumerate(view_cams):
                if count_manual(a2.get(cam, {}), names) < args.min_view_manual:
                    continue  # not manually labelled in this camera -> skip the view
                if k == ego_view:
                    ext = ego_extr.get(str(fid))
                    if ext is None:
                        continue
                    ext = np.asarray(ext, dtype=np.float64)
                    Rk, tk = ext[:, :3], ext[:, 3]
                else:
                    Rk, tk = Rwc[k], twc[k]
                w, h = int(img_size[k][0]), int(img_size[k][1])
                pc = (Rk @ world.T).T + tk            # (21,3) camera frame
                j2d = project(K[k], pc)
                if not view_sees_hand(j2d, pc[:, 2], mask, w, h):
                    continue
                vconf = mask.copy()
                vconf[pc[:, 2] <= 0] = 0.0
                bbox = bbox_from_joints2d(j2d, vconf, w, h)
                if bbox is None:
                    continue
                entry = {
                    "bbox": bbox,
                    "joints_2d": j2d.astype(np.float32),
                    "joints_3d": pc.astype(np.float32),
                    "joints_conf": vconf.astype(np.float32),
                }
                if k == ego_view:
                    entry["extrinsics"] = np.concatenate(
                        [Rk, tk[:, None]], axis=1
                    ).astype(np.float32)
                views[k] = entry
                needed[k].add(fid)

            if views:
                hands.append({"hand_id": hand_id, "side": side,
                              "has_mano": False, "views": views})
        if hands:
            frames.append({"frame_id": fid, "is_annotated": True, "hands": hands})
    return frames, needed


# ── image pass (decode videos) ───────────────────────────────────────────────────

def get_aria_calibs(vrs_path):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    src = provider.get_device_calibration().get_camera_calib("camera-rgb")
    src = src.rescale([ARIA_RGB, ARIA_RGB], ARIA_RGB / ARIA_SENSOR)
    dst = calibration.get_linear_camera_calibration(
        ARIA_UNDIST, ARIA_UNDIST, ARIA_FOCAL, "camera-rgb"
    )
    return src, dst


def extract_video_frames(video_path, needed_ids, processor):
    """Decode `video_path` sequentially; for each frame index in `needed_ids` call
    processor(av_frame)->BGR ndarray. Returns {frame_id: bgr}."""
    out = {}
    if not needed_ids:
        return out
    maxt = max(needed_ids)
    try:
        cont = av.open(video_path)
    except Exception as e:  # noqa
        print(f"[warn] cannot open {video_path}: {e}", file=sys.stderr)
        return out
    idx = 0
    for fr in cont.decode(video=0):
        if idx in needed_ids:
            try:
                out[idx] = processor(fr)
            except Exception as e:  # noqa
                print(f"[warn] decode {video_path}@{idx}: {e}", file=sys.stderr)
        if idx >= maxt:
            break
        idx += 1
    cont.close()
    return out


def write_images(take, calib, view_cams, ego_view, needed, sess_dir, data_dir):
    """Decode each view's video, write needed JPEGs. Returns {view: set(frame_ids) written}."""
    fav = take["frame_aligned_videos"]
    root = take["root_dir"]
    written = {k: set() for k in range(len(view_cams))}

    aria_calibs = None
    if ego_view is not None and needed[ego_view]:
        vrs = os.path.join(data_dir, root, f"{view_cams[ego_view]}_noimagestreams.vrs")
        if os.path.exists(vrs):
            try:
                aria_calibs = get_aria_calibs(vrs)
            except Exception as e:  # noqa
                print(f"[warn] aria calib {vrs}: {e}", file=sys.stderr)

    for k, cam in enumerate(view_cams):
        if not needed[k]:
            continue
        if k == ego_view:
            if aria_calibs is None:
                continue
            src, dst = aria_calibs
            rel = fav[cam]["rgb"]["relative_path"]

            def proc(fr, _src=src, _dst=dst):
                pil = fr.to_image().rotate(90, expand=True)    # CCW, upright (matches yolo26)
                rect = calibration.distort_by_calibration(np.asarray(pil), _dst, _src)
                return cv2.cvtColor(np.asarray(rect), cv2.COLOR_RGB2BGR)
        else:
            rel = exo_relpath(fav[cam])

            def proc(fr):
                return fr.to_ndarray(format="bgr24")

        video_path = os.path.join(data_dir, root, rel)
        frames = extract_video_frames(video_path, needed[k], proc)
        out_dir = os.path.join(sess_dir, "images", f"view{k}")
        os.makedirs(out_dir, exist_ok=True)
        for fid, bgr in frames.items():
            if cv2.imwrite(os.path.join(out_dir, f"{fid:06d}.jpg"), bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
                written[k].add(fid)
    return written


# ── session driver ───────────────────────────────────────────────────────────────

def session_complete(sess_dir):
    return (os.path.exists(os.path.join(sess_dir, "calib.npz"))
            and os.path.isdir(os.path.join(sess_dir, "frames"))
            and len(os.listdir(os.path.join(sess_dir, "frames"))) > 0)


def convert_take(take, split, hand_ann_path, data_dir, out_dir, args):
    cam_pose_path = os.path.join(
        data_dir, "annotations", "ego_pose", split, "camera_pose",
        os.path.basename(hand_ann_path),
    )
    if not os.path.exists(cam_pose_path):
        return take["take_name"], 0, 0, "no camera_pose"
    sess_dir = os.path.join(out_dir, split, take["take_name"])
    if not args.overwrite and session_complete(sess_dir):
        return take["take_name"], 0, 0, "skip"

    hand_ann = json.load(open(hand_ann_path))
    cam_pose = json.load(open(cam_pose_path))
    ego = ego_cam_id(take)
    has_ego_media = (
        ego is not None and ego in take["frame_aligned_videos"]
        and "rgb" in take["frame_aligned_videos"][ego]
        and os.path.exists(os.path.join(
            data_dir, take["root_dir"],
            take["frame_aligned_videos"][ego]["rgb"]["relative_path"]))
    )

    calib, view_cams, ego_view, ego_extr = build_views(
        take, hand_ann, cam_pose, data_dir, has_ego_media)
    if len(view_cams) == 0:
        return take["take_name"], 0, 0, "no views/media"

    frames, needed = convert_geometry(
        take, hand_ann, cam_pose, calib, view_cams, ego_view, ego_extr, args)
    if not frames:
        return take["take_name"], 0, 0, "no labelled hands"

    os.makedirs(sess_dir, exist_ok=True)
    np.savez(os.path.join(sess_dir, "calib.npz"), **calib)

    if args.skip_images:
        written = needed
    else:
        written = write_images(take, calib, view_cams, ego_view, needed, sess_dir, data_dir)

    # Prune view entries whose image failed to write, then drop empty hands/frames so the
    # dataloader never references a missing JPEG.
    frames_dir = os.path.join(sess_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    n_frames = n_hands = 0
    for fr in frames:
        kept_hands = []
        for hand in fr["hands"]:
            vws = {k: v for k, v in hand["views"].items() if fr["frame_id"] in written[k]}
            if vws:
                hand["views"] = vws
                kept_hands.append(hand)
        if not kept_hands:
            continue
        fr["hands"] = kept_hands
        np.savez(os.path.join(frames_dir, f"{fr['frame_id']:06d}.npz"),
                 data=np.array(fr, dtype=object))
        n_frames += 1
        n_hands += len(kept_hands)
    return take["take_name"], n_frames, n_hands, None


def _worker(arg):
    take, split, path, data_dir, out_dir, args = arg
    try:
        return convert_take(take, split, path, data_dir, out_dir, args)
    except Exception as e:  # noqa  keep pool alive
        return take["take_name"], 0, 0, f"{type(e).__name__}: {e}"


def discover(data_dir):
    takes = json.load(open(os.path.join(data_dir, "takes.json")))
    by_uid = {t["take_uid"]: t for t in takes}
    work = []
    for split in SPLITS:
        ann_dir = os.path.join(data_dir, "annotations", "ego_pose", split, "hand", "annotation")
        if not os.path.isdir(ann_dir):
            continue
        for f in sorted(os.listdir(ann_dir)):
            if not f.endswith(".json"):
                continue
            t = by_uid.get(f[:-5])
            if t is not None:
                work.append((t, split, os.path.join(ann_dir, f)))
    return work


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N takes (smoke test)")
    ap.add_argument("--debug", action="store_true",
                    help="process only the first few takes (sets --limit 3 if unset)")
    ap.add_argument("--min-view-manual", type=int, default=4,
                    help="min manually-placed 2D keypoints a camera must have for a hand to "
                         "keep that view (drops auto-only views)")
    ap.add_argument("--min-joints", type=int, default=6,
                    help="min triangulated 3D joints (incl. wrist) to keep a hand")
    ap.add_argument("--skip-images", action="store_true", help="geometry/npz only (debug)")
    ap.add_argument("--overwrite", action="store_true", help="re-convert complete sessions")
    args = ap.parse_args()

    if args.debug and not args.limit:
        args.limit = 3
    work = discover(args.data_dir)
    if args.limit:
        work = work[: args.limit]
    print(f"Converting {len(work)} takes -> {args.out_dir} ({args.workers} workers)")
    for split in SPLITS:
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)

    tasks = [(t, split, path, args.data_dir, args.out_dir, args) for (t, split, path) in work]
    grand = {"takes": 0, "skipped": 0, "failed": 0, "frames": 0, "hands": 0}

    def tally(name, nf, nh, err):
        grand["takes"] += 1
        if err == "skip":
            grand["skipped"] += 1
        elif err is not None:
            grand["failed"] += 1
            tqdm.write(f"  {name}: {err}")
        else:
            grand["frames"] += nf; grand["hands"] += nh
            tqdm.write(f"  {name}: {nf} frames, {nh} hand-instances")

    if args.workers <= 1:
        for t in tqdm(tasks, desc="takes"):
            tally(*_worker(t))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for res in tqdm(pool.imap_unordered(_worker, tasks),
                            total=len(tasks), desc="takes"):
                tally(*res)

    print(f"\nDone. {grand['takes']} takes ({grand['skipped']} skipped, "
          f"{grand['failed']} failed), {grand['frames']} frames, "
          f"{grand['hands']} hand-instances -> {args.out_dir}")


if __name__ == "__main__":
    main()

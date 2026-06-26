"""
SessionDataset: map-style dataset over our own multi-view / multi-hand intermediate format.

On-disk layout of one session (see SESSION_DATALOADER.md):

    session_xxx/
        calib.npz                 # per-view pinhole params (static); view 0 == world frame
        frames/000001.npz         # one logical sample == one multi-view, multi-hand frame
        images/view0/000001.jpg
        images/view1/000001.jpg
        ...

calib.npz arrays are indexed by view:
    K              (V, 3, 3)
    dist           (V, D)
    R_world_to_cam (V, 3, 3)      # view 0 is identity (world frame)
    t_world_to_cam (V, 3)         # view 0 is zeros
    img_size       (V, 2)         # [W, H] per view
    dynamic_views  (M,) int       # OPTIONAL. View indices whose extrinsics vary per frame
                                  #   (ego/head-mounted cameras). For these the calib R/t are a
                                  #   placeholder (used only for empty samples); the real per-frame
                                  #   world->cam lives in the frame's per-view "extrinsics" entry.
                                  #   Absent on purely-static sessions.

frames/NNNNNN.npz:
    frame_id : int
    hands    : list of dicts (stored object array), each:
        {
            "hand_id": int,
            "side":    "right" | "left",
            "mano":    {"global_orient": (3,) axis-angle WORLD frame,
                        "hand_pose": (45,), "betas": (10,)},
            "views":   {view_idx: {"bbox": (4,), "joints_2d": (21,2),
                                   "joints_3d": (21,3),
                                   # for dynamic_views only: per-frame world->cam [R|t]
                                   ["extrinsics": (3,4), "K": (3,3), "trans": (3,)]}, ...}
        }

__getitem__(idx) returns ONE frame as a list of NUM_VIEWS view-groups; each view-group is a
list of per-hand sample dicts (one per hand that this view sees). View-groups with no hand are
a single zero-filled dict with valid=False. See `session_collate` for how this batches.

`bad_left_views`/`bad_right_views` (constructor args, set per-dataset from config
BAD_LEFT_VIEWS/BAD_RIGHT_VIEWS; NOT stored on disk) are runtime filters of low-quality view
indices, applied per hand side -- occlusion is often side-dependent (e.g. a side camera blocked
by the body for the far hand but not the near one). A hand of that side in those views is
skipped before any image is loaded, so the view only ever yields the masked empty placeholder
and is excluded by `select_views`. Default empty -> no-op.

The per-hand dict mirrors the field schema of
`src/metric_hand_tracking/wilor/datasets/vitdet_dataset.py` (img, personid, box_center,
box_size, img_size, right) plus training GT (theta/mano_params, betas, joints_3d in camera
frame, joints_2d in crop frame, extrinsics world->cam, valid).
"""
import os
import glob
import json
import hashlib
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from yacs.config import CfgNode

from src.metric_hand_tracking.wilor.datasets.utils import get_example, expand_to_aspect_ratio

# Mirror the constants defined in image_dataset.py (kept local so this module does not pull in
# webdataset/braceexpand, which image_dataset.py imports at module level).
FLIP_KEYPOINT_PERMUTATION = list(range(21))
DEFAULT_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255.0 * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256

NUM_KEYPOINTS = 21

# All session datasets store joints in HO-3D / raw-smplx-MANO order:
#   wrist, index(1-3), middle(1-3), pinky(1-3), ring(1-3), thumb(1-4),
#   index_tip, middle_tip, ring_tip, pinky_tip
# WiLoR predicts in OpenPose order:
#   wrist, thumb(cmc-tip), index(mcp-tip), middle(mcp-tip), ring(mcp-tip), pinky(mcp-tip)
# This permutation (== mano_to_openpose in mano_wrapper.py) converts stored → OpenPose.
_STORED_TO_OPENPOSE = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)


def _crop_correction_rotation(
    box_center: np.ndarray, focal_length: float, img_center: np.ndarray
) -> np.ndarray:
    """3x3 rotation R that aligns the camera +Z axis (0,0,1) with the ray to the crop center.

    NumPy port of ``hand_geometry.get_crop_correction_rotation`` for a single box, kept local
    so the dataloader does not import the pyrender-heavy ``hand_geometry`` module into every
    worker. Verified to match the canonical implementation to float32 precision.

    Args:
        box_center:   (2,) crop center (u, v) in full-image pixels.
        focal_length: scalar camera focal length (K[0, 0]).
        img_center:   (2,) principal point (K[0, 2], K[1, 2]).
    Returns:
        (3, 3) float32 rotation matrix.
    """
    dx = float(box_center[0]) - float(img_center[0])
    dy = float(box_center[1]) - float(img_center[1])
    ray = np.array([dx, dy, float(focal_length)], dtype=np.float64)
    ray /= np.linalg.norm(ray)

    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, ray)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z_axis, ray), -1.0, 1.0))
    angle = np.arccos(dot)
    axis_angle = (axis / axis_norm) * angle if axis_norm > 1e-6 else np.zeros(3)

    return Rot.from_rotvec(axis_angle).as_matrix().astype(np.float32)


class SessionDataset:
    """Map-style dataset over a single `session_xxx/` directory."""

    def __init__(self, cfg: CfgNode, path: str, train: bool = True,
                 frame_files: Optional[List[str]] = None,
                 bad_left_views: Optional[List[int]] = None,
                 bad_right_views: Optional[List[int]] = None, **kwargs):
        self.cfg = cfg
        self.path = path
        self.train = train

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
        self.bbox_shape = cfg.MODEL.get("BBOX_SHAPE", None)
        self.bbox_scale = float(cfg.DATASETS.get("BBOX_SCALE", 1.0))

        # --- load static calibration ---
        calib = np.load(os.path.join(path, "calib.npz"), allow_pickle=True)
        self.K = np.asarray(calib["K"], dtype=np.float32)                  # (V,3,3)
        self.dist = np.asarray(calib["dist"], dtype=np.float32) if "dist" in calib else None
        self.R_w2c = np.asarray(calib["R_world_to_cam"], dtype=np.float32)  # (V,3,3)
        self.t_w2c = np.asarray(calib["t_world_to_cam"], dtype=np.float32)  # (V,3)
        self.calib_img_size = np.asarray(calib["img_size"], dtype=np.float32)  # (V,2) [W,H]
        self.num_views = self.K.shape[0]
        # Views whose extrinsics vary per frame (ego/head-mounted cameras). For these the
        # per-frame world->cam is read from the frame's per-view "extrinsics" entry; the static
        # calib R/t above are only a placeholder used by _empty_sample (valid=False). Static
        # sessions omit this key -> empty set -> behavior is byte-for-byte unchanged.
        self.dynamic_views = (
            {int(v) for v in calib["dynamic_views"]} if "dynamic_views" in calib else set()
        )
        # Runtime dataloader filter (NOT stored on disk): per-side view indices to drop for this
        # dataset because the camera is low quality / occludes that hand specifically (occlusion
        # can be side-dependent, e.g. a camera blocked by the body for only one hand). A hand of
        # that side in these views is never emitted as a valid sample -- it falls through to
        # _empty_sample (valid=False) and is excluded by select_views, so it costs no image IO,
        # forward pass, or gradient. Empty sets = no-op.
        self.bad_left_views = {int(v) for v in (bad_left_views or [])}
        self.bad_right_views = {int(v) for v in (bad_right_views or [])}

        # --- index frames ---
        if frame_files is not None:
            # Fast path: caller (MixedSessionDataset index cache) pre-computed the frame list.
            self.frame_files = frame_files
        else:
            raw_files = sorted(glob.glob(os.path.join(path, "frames", "*.npz")))
            if len(raw_files) == 0:
                raise FileNotFoundError(f"No frames found under {os.path.join(path, 'frames')}")
            # Drop frames with no annotated hands (e.g. dexycb's leading unannotated frames) so
            # we never sample empty all-padding frames. Disable with DATASETS.FILTER_EMPTY_FRAMES:
            # False. The kept-frame list is cached per-session; disable with CACHE_FRAME_INDEX:
            # False.
            if cfg.DATASETS.get("FILTER_EMPTY_FRAMES", True):
                raw_files = self._filter_annotated(
                    raw_files, use_cache=cfg.DATASETS.get("CACHE_FRAME_INDEX", True)
                )
                if len(raw_files) == 0:
                    raise FileNotFoundError(
                        f"No annotated frames (with >=1 hand) under {os.path.join(path, 'frames')}"
                    )
            self.frame_files = raw_files

    def __len__(self) -> int:
        return len(self.frame_files)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_frame(frame_file: str) -> Dict[str, Any]:
        """frames/*.npz wrap everything in a single pickled `data` dict."""
        return np.load(frame_file, allow_pickle=True)["data"].item()

    @staticmethod
    def _frame_has_hands(frame_file: str) -> bool:
        return len(SessionDataset._load_frame(frame_file).get("hands", [])) > 0

    CACHE_NAME = ".kept_frames_cache.json"

    def _filter_annotated(self, frame_files: List[str], use_cache: bool = True) -> List[str]:
        """Return only frames with >=1 hand, caching the result per session.

        The cache is keyed by a fingerprint of the full frame set (sorted basenames), so it is
        transparently rebuilt if frames are added/removed/renamed. A read-only session dir simply
        skips writing the cache (the scan still runs).
        """
        basenames = [os.path.basename(f) for f in frame_files]
        fingerprint = hashlib.md5("\n".join(basenames).encode()).hexdigest()
        cache_path = os.path.join(self.path, self.CACHE_NAME)

        if use_cache and os.path.isfile(cache_path):
            try:
                with open(cache_path) as fh:
                    cached = json.load(fh)
                if cached.get("fingerprint") == fingerprint:
                    kept = set(cached["kept"])
                    return [f for f, b in zip(frame_files, basenames) if b in kept]
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt/stale cache -> rescan

        _iter = zip(frame_files, basenames)
        if _tqdm is not None:
            _iter = _tqdm(
                _iter, total=len(frame_files),
                desc=f"Scanning {os.path.basename(self.path)}",
                unit="fr", leave=False,
            )
        kept_files = [f for f, b in _iter if self._frame_has_hands(f)]

        if use_cache:
            kept_basenames = [os.path.basename(f) for f in kept_files]
            tmp = cache_path + f".tmp.{os.getpid()}"
            try:
                with open(tmp, "w") as fh:
                    json.dump({"fingerprint": fingerprint, "kept": kept_basenames}, fh)
                os.replace(tmp, cache_path)  # atomic; safe across workers
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass  # read-only session dir -> skip caching
        return kept_files

    def _image_path(self, view: int, frame_id: int) -> str:
        return os.path.join(self.path, "images", f"view{view}", f"{frame_id:06d}.jpg")

    def _empty_sample(self, view: int) -> Dict[str, Any]:
        """Zero-filled placeholder for a view that sees no hand (valid=False)."""
        s = self.img_size
        return {
            "img": np.zeros((3, s, s), dtype=np.float32),
            "right": np.float32(1.0),
            "mano_params": {
                "global_orient": np.zeros(3, dtype=np.float32),
                "hand_pose": np.zeros(45, dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
            },
            "has_mano": np.float32(0.0),
            "keypoints_3d": np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32),
            "keypoints_2d": np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32),
            "extrinsics": np.concatenate(
                [self.R_w2c[view], self.t_w2c[view][:, None]], axis=1
            ).astype(np.float32),
            "R_crop_correction": np.eye(3, dtype=np.float32),
            "valid": False,
            "hand_id": -1,
        }

    def _world_to_cam_global_orient(
        self, global_orient_world: np.ndarray, R_w2c: np.ndarray
    ) -> np.ndarray:
        """Compose world-frame root rotation with the view extrinsic: R_w2c @ exp(go_world)."""
        R_world = Rot.from_rotvec(global_orient_world.reshape(3))
        R_cam = Rot.from_matrix(R_w2c) * R_world
        return R_cam.as_rotvec().astype(np.float32)

    def _build_hand_view_sample(self, hand: Dict, view: int, frame_id: int) -> Dict[str, Any]:
        vobs = hand["views"][view]
        side = hand["side"]
        is_right = side == "right"
        # Keypoint-only datasets (e.g. egoexo4d) carry no MANO fit. Such hands set
        # "has_mano": False and may omit "mano"; we feed zero placeholder params and flag
        # has_mano_params=0 so the MANO parameter losses are masked out (the 2D/3D keypoint
        # losses still supervise). Datasets with MANO leave has_mano True (default) -> unchanged.
        has_mano = bool(hand.get("has_mano", True))
        mano = hand.get("mano") or {
            "global_orient": np.zeros(3, dtype=np.float32),
            "hand_pose": np.zeros(45, dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
        }

        # --- effective world->cam extrinsics for THIS (view, frame) ---
        # Dynamic (ego) views carry their per-frame extrinsics in the frame's per-view entry;
        # static views fall back to the session-level calib.
        if view in self.dynamic_views and "extrinsics" in vobs:
            ext = np.asarray(vobs["extrinsics"], dtype=np.float32)  # (3,4) world->cam [R|t]
            R_w2c_eff, t_w2c_eff = ext[:, :3], ext[:, 3]
        else:
            R_w2c_eff, t_w2c_eff = self.R_w2c[view], self.t_w2c[view]

        # --- camera-frame global orient ---
        global_orient_cam = self._world_to_cam_global_orient(
            np.asarray(mano["global_orient"], dtype=np.float32), R_w2c_eff
        )
        hand_pose = np.asarray(mano["hand_pose"], dtype=np.float32).reshape(-1)
        betas = np.asarray(mano["betas"], dtype=np.float32).reshape(-1)

        mano_params = {
            "global_orient": global_orient_cam,
            "hand_pose": hand_pose,
            "betas": betas,
        }
        _hm = np.array(1.0 if has_mano else 0.0)
        has_mano_params = {
            "global_orient": _hm,
            "hand_pose": _hm,
            "betas": _hm,
        }

        # --- per-joint confidence (mask) ---
        # Datasets with partial GT (e.g. egoexo4d, where some joints fail triangulation) store a
        # (21,) "joints_conf" of 1/0 per joint; the keypoint 2D/3D losses use it as a per-joint
        # mask so missing joints aren't supervised. Absent -> all ones (fully-labelled hands).
        joints_conf = np.asarray(
            vobs.get("joints_conf", np.ones(NUM_KEYPOINTS, dtype=np.float32)), dtype=np.float32
        ).reshape(NUM_KEYPOINTS)
        joints_conf = joints_conf[_STORED_TO_OPENPOSE]  # HO-3D order -> OpenPose

        # --- 3D joints in CAMERA frame ---
        joints_3d = np.asarray(vobs["joints_3d"], dtype=np.float32)  # (21,3)
        if joints_3d.shape == (NUM_KEYPOINTS, 3) and self.cfg.DATASETS.get(
            "JOINTS3D_IN_WORLD", False
        ):
            # stored in world frame -> transform: X_cam = R @ X_world + t
            joints_3d = joints_3d @ R_w2c_eff.T + t_w2c_eff[None, :]
        joints_3d = joints_3d[_STORED_TO_OPENPOSE]  # HO-3D order -> OpenPose
        keypoints_3d = np.concatenate(
            [joints_3d, joints_conf[:, None]], axis=1
        )  # (21,4) with conf

        # --- 2D joints (full-image px) with conf ---
        joints_2d = np.asarray(vobs["joints_2d"], dtype=np.float32)  # (21,2)
        joints_2d = joints_2d[_STORED_TO_OPENPOSE]   # HO-3D order -> OpenPose
        keypoints_2d = np.concatenate(
            [joints_2d, joints_conf[:, None]], axis=1
        )  # (21,3)

        # --- bbox -> center/scale (matches image_dataset.py) ---
        bbox = np.asarray(vobs["bbox"], dtype=np.float32)  # (x1,y1,x2,y2)
        center = (bbox[2:4] + bbox[0:2]) / 2.0
        scale = (bbox[2:4] - bbox[0:2]) / 200.0
        bbox_size = expand_to_aspect_ratio(
            scale * 200, target_aspect_ratio=self.bbox_shape
        ).max() * self.bbox_scale

        # --- crop-correction rotation (aligns +Z with the ray to the crop center) ---
        # Computed from the original full-image crop center; used downstream to undo the
        # implicit rotation a centered crop induces on the 3D joints.
        K_view = self.K[view]
        R_crop_correction = _crop_correction_rotation(
            center, K_view[0, 0], K_view[:2, 2]
        )

        # --- CLIFF-style crop-frame correction (optional, DATASETS.APPLY_CROP_CORRECTION) ---
        # WiLoR predicts in a crop-centered pinhole camera (principal point == crop center, see
        # MultiViewWiLoR._predicted_K). A crop whose center is off the image principal point
        # therefore corresponds to a camera tilted by R_crop_correction. To supervise in the SAME
        # frame the network predicts in, rotate the GT root orientation AND the 3D joints by
        # R_crop_correction^T. The 2D keypoints are literal crop pixels and stay unchanged.
        # Applied BEFORE get_example so the flip / in-plane-rot augmentation composes identically
        # over both signals (both are isometries of the 3D structure; the mirror conjugation and
        # the Z-rotation commute with the correction). The 3D loss is root-relative (pelvis_id=0),
        # so the joints are rotated about the wrist (joint 0); absolute translation is irrelevant.
        # At inference, recover camera/world orientation with R_cam = R_crop_correction @ R_pred.
        if self.cfg.DATASETS.get("APPLY_CROP_CORRECTION", False):
            Cc = R_crop_correction  # (3,3) maps crop-camera +Z onto the crop-center ray
            go_mat = Rot.from_rotvec(
                np.asarray(mano_params["global_orient"], dtype=np.float64).reshape(3)
            ).as_matrix()
            mano_params["global_orient"] = (
                Rot.from_matrix(Cc.T @ go_mat).as_rotvec().astype(np.float32)
            )
            root = keypoints_3d[0, :3].copy()
            keypoints_3d[:, :3] = (keypoints_3d[:, :3] - root) @ Cc + root

        # --- load image & crop/augment via get_example ---
        img_path = self._image_path(view, frame_id)
        (
            img_patch,
            keypoints_2d,
            keypoints_3d,
            mano_params,
            has_mano_params,
            img_size,
            trans,
        ) = get_example(
            img_path,
            center[0], center[1],
            bbox_size, bbox_size,
            keypoints_2d, keypoints_3d,
            mano_params, has_mano_params,
            FLIP_KEYPOINT_PERMUTATION,
            self.img_size, self.img_size,
            self.mean, self.std,
            self.train, is_right, self.cfg.DATASETS.CONFIG,
            is_bgr=True, return_trans=True,
        )

        return {
            "img": img_patch,
            "right": np.float32(1.0 if is_right else 0.0),
            # --- training GT ---
            "mano_params": mano_params,
            "keypoints_3d": keypoints_3d.astype(np.float32),  # camera frame, (21,4)
            "keypoints_2d": keypoints_2d.astype(np.float32),  # crop frame, (21,3)
            # scalar MANO-GT mask, forwarded by session_collate so the multiview loss can mask
            # the MANO-parameter terms for keypoint-only hands (egoexo4d).
            "has_mano": np.float32(1.0 if has_mano else 0.0),
            "extrinsics": np.concatenate(
                [R_w2c_eff, t_w2c_eff[:, None]], axis=1
            ).astype(np.float32),  # (3,4) world->cam
            "R_crop_correction": R_crop_correction,  # (3,3) crop-center ray alignment
            "valid": True,
            "hand_id": int(hand["hand_id"]),
        }

    # ------------------------------------------------------------------ main
    def __getitem__(self, idx: int) -> List[List[Dict[str, Any]]]:
        frame = self._load_frame(self.frame_files[idx])  # {frame_id, is_annotated, hands: [...]}
        frame_id = int(frame["frame_id"])
        hands = list(frame["hands"])  # list of hand dicts

        view_groups: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_views)]
        for hand in hands:
            views = hand["views"]
            bad_views = self.bad_right_views if hand["side"] == "right" else self.bad_left_views
            for view in views.keys():
                view = int(view)
                if view in bad_views:
                    continue  # bad view for this hand side: never emit a valid sample
                view_groups[view].append(
                    self._build_hand_view_sample(hand, view, frame_id)
                )

        # zero-fill empty view slots so every frame yields NUM_VIEWS groups
        for view in range(self.num_views):
            if len(view_groups[view]) == 0:
                view_groups[view].append(self._empty_sample(view))

        return view_groups

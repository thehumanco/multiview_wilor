"""
Mixing + batching for SessionDataset.

- `session_collate`: turns a batch of frames (each a list of NUM_VIEWS view-groups, each a
  variable-length list of per-hand dicts) into a dict keyed by view, with all hands stacked
  across the batch and a `valid` mask. This is what feeds MultiViewWiLoR's view sampling /
  gather step.
- `MixedSessionDataset`: concatenates per-session `SessionDataset`s and exposes a
  `WeightedRandomSampler` so sessions are sampled by their configured proportions (which must
  sum to 1) regardless of raw session size -- the map-style analogue of `wds.RandomMix`.
- `SessionDataModule`: LightningDataModule for the session-format datasets.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from yacs.config import CfgNode

from .session_dataset import SessionDataset

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

_INDEX_CACHE_DIR = Path(__file__).resolve().parents[1] / "dataset_caches"

WEIGHT_SUM_TOL = 1e-6


def _index_fingerprint(
    session_lists: List[List[str]],
    weights: np.ndarray,
    train: bool,
    filter_empty: bool,
) -> str:
    parts = ["train" if train else "val", str(filter_empty)]
    for sessions, w in zip(session_lists, weights):
        parts.append(f"w={w:.8f}")
        parts.extend(sessions)  # discover_sessions returns sorted paths
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def _load_index_cache(fp: str) -> Optional[Dict[str, List[str]]]:
    """Return {session_path: [frame_basename, ...]} or None on miss."""
    p = _INDEX_CACHE_DIR / f"{fp}.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        if data.get("fingerprint") == fp:
            return data["sessions"]
    except Exception:
        pass
    return None


def _save_index_cache(fp: str, sessions: Dict[str, List[str]]) -> None:
    """Atomically write {session_path: [frame_basename, ...]} to the index cache."""
    try:
        _INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _INDEX_CACHE_DIR / f"{fp}.json"
        tmp = str(p) + f".{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"fingerprint": fp, "sessions": sessions}, f)
        os.replace(tmp, str(p))
    except OSError:
        pass  # read-only or permission issue -> skip caching


def _is_session(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "calib.npz"))


def discover_sessions(root: str, max_depth: int = 3) -> List[str]:
    """Expand a dataset-root path into its session directories.

    A directory is a *session* if it contains `calib.npz`. Handles a session given directly, a
    flat root of sessions (oakink `*_sessions/<scene>/`), and split roots
    (ho3d `ho3d_sessions/train/<seq>/`, interhand `interhand_sessions/train/<capture>/`).

    Uses a bounded breadth-first walk (not `**` recursive glob) so it stays fast over NFS:
    once a directory qualifies as a session we do not descend into it.
    """
    root = os.path.expanduser(os.path.expandvars(root))
    sessions: List[str] = []
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop()
        if _is_session(d):
            sessions.append(d)
            continue  # don't descend into a session
        if depth >= max_depth:
            continue
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir():
                    frontier.append((e.path, depth + 1))
    if not sessions:
        raise FileNotFoundError(f"No sessions (calib.npz) found under {root} (max_depth={max_depth})")
    return sorted(sessions)


# --------------------------------------------------------------------- collate
def _stack(values: List[Any]) -> Any:
    """Stack a list of per-hand field values into a single tensor."""
    arr = np.stack([np.asarray(v) for v in values], axis=0)
    return torch.from_numpy(arr)


def session_collate(batch: List[List[List[Dict[str, Any]]]]) -> Dict[int, Dict[str, Any]]:
    """
    Args:
        batch: list (len B) of frames; each frame is a list (len V) of view-groups;
               each view-group is a list of per-hand dicts.
    Returns:
        {view_idx: {field: stacked tensor over all hands in all frames for that view, ...,
                    'frame_index': (N,) which batch frame each hand came from,
                    'valid': (N,) bool}}
    """
    # Frames can have different view counts (single-view sessions, oakink=3, ho3d=5, ...),
    # so iterate over the MAX view count; frames lacking a given view contribute no hands.
    # (min() here silently collapses every mixed batch to view 0 -> select_views returns [].)
    num_views = max(len(frame) for frame in batch)
    out: Dict[int, Dict[str, Any]] = {}

    # scalar/array fields to stack (mano_params handled separately)
    tensor_fields = [
        "img", "right", "keypoints_3d", "keypoints_2d",
        "extrinsics", "R_crop_correction", "has_mano",
    ]
    meta_fields = ["hand_id", "valid"]

    for view in range(num_views):
        # gather every hand-dict for this view across the batch
        hand_dicts: List[Dict[str, Any]] = []
        frame_index: List[int] = []
        for b, frame in enumerate(batch):
            if view >= len(frame):
                continue  # this frame's dataset has fewer views than `view`
            for hand in frame[view]:
                hand_dicts.append(hand)
                frame_index.append(b)

        if not hand_dicts:
            continue  # no frame in this batch has this view

        view_out: Dict[str, Any] = {}
        for f in tensor_fields:
            view_out[f] = _stack([h[f] for h in hand_dicts])
        # mano params: split tensors
        view_out["mano_params"] = {
            k: _stack([h["mano_params"][k] for h in hand_dicts])
            for k in ("global_orient", "hand_pose", "betas")
        }
        for f in meta_fields:
            if f == "valid":
                view_out["valid"] = torch.tensor(
                    [bool(h["valid"]) for h in hand_dicts], dtype=torch.bool
                )
            else:
                view_out[f] = torch.tensor([int(h[f]) for h in hand_dicts])
        view_out["frame_index"] = torch.tensor(frame_index, dtype=torch.long)
        out[view] = view_out

    return out


# ----------------------------------------------------------------- mixed dataset
class MixedSessionDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset over multiple datasets, each a root path of sessions.

    Each config entry is one *dataset* (`PATH` = a root containing many sessions, e.g.
    `.../oakink/oakink_sessions`) with a `WEIGHT` = that dataset's overall sampling proportion.
    WEIGHTs across entries must sum to 1.0. Within a dataset the proportion is spread uniformly
    over its frames, so a session's contribution scales with its frame count — the map-style
    analogue of `wds.RandomMix`.
    """

    def __init__(self, cfg: CfgNode, train: bool = True) -> None:
        entries = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        weights = np.array([float(e["WEIGHT"]) for e in entries], dtype=np.float64)
        if abs(weights.sum() - 1.0) > WEIGHT_SUM_TOL:
            raise ValueError(
                f"Dataset WEIGHTs must sum to 1.0 (got {weights.sum():.6f}); "
                f"check the data config."
            )

        use_cache = cfg.DATASETS.get("CACHE_FRAME_INDEX", True)
        filter_empty = cfg.DATASETS.get("FILTER_EMPTY_FRAMES", True)
        split = "train" if train else "val"

        # Discover sessions (fast BFS - just checks for calib.npz).
        session_lists = [discover_sessions(e["PATH"]) for e in entries]
        n_total = sum(len(sl) for sl in session_lists)

        # Check the top-level index cache: maps session_path -> [frame_basename, ...].
        # A cache hit lets us skip glob + _filter_annotated for every session.
        index_cache: Optional[Dict[str, List[str]]] = None
        fingerprint: Optional[str] = None
        if use_cache:
            fingerprint = _index_fingerprint(session_lists, weights, train, filter_empty)
            index_cache = _load_index_cache(fingerprint)
            if index_cache is not None:
                print(f"[MixedSessionDataset] {split} index cache hit "
                      f"({len(index_cache)} sessions cached, {n_total} total)")

        # Build SessionDataset objects.  Show a tqdm bar on the slow (cache-miss) path.
        datasets: List[SessionDataset] = []
        sample_weights: List[np.ndarray] = []
        new_entries: Dict[str, List[str]] = {}  # sessions built without top-level cache

        bar = (
            _tqdm(total=n_total, desc=f"Indexing {split} sessions", unit="session")
            if _tqdm is not None and index_cache is None
            else None
        )

        for entry, w, sessions in zip(entries, weights, session_lists):
            entry_datasets: List[SessionDataset] = []
            # per-dataset, per-side low-quality views to drop (occlusion can depend on hand side)
            bad_left_views = entry.get("BAD_LEFT_VIEWS", [])
            bad_right_views = entry.get("BAD_RIGHT_VIEWS", [])
            for p in sessions:
                try:
                    if index_cache is not None and p in index_cache:
                        ff = [os.path.join(p, "frames", b) for b in index_cache[p]]
                        ds = SessionDataset(cfg, p, train=train, frame_files=ff,
                                             bad_left_views=bad_left_views,
                                             bad_right_views=bad_right_views)
                    else:
                        ds = SessionDataset(cfg, p, train=train,
                                             bad_left_views=bad_left_views,
                                             bad_right_views=bad_right_views)
                        if use_cache:
                            new_entries[p] = [os.path.basename(f) for f in ds.frame_files]
                except FileNotFoundError as e:
                    # Degenerate session: no frames, or no frame with >=1 annotated hand (e.g. an
                    # InterHand sequence whose every frame is hand_type_valid=0 -> converter emits
                    # only empty-`hands` frames). Auto-discovery legitimately surfaces these; skip
                    # rather than killing the whole run.
                    print(f"[MixedSessionDataset] skipping empty session {p}: {e}")
                    if bar is not None:
                        bar.update(1)
                    continue
                entry_datasets.append(ds)
                if bar is not None:
                    bar.update(1)

            n_frames = sum(len(ds) for ds in entry_datasets)
            if n_frames == 0:
                raise FileNotFoundError(f"Dataset {entry['PATH']} has 0 frames")
            # Spread this dataset's proportion w uniformly over all its frames.
            for ds in entry_datasets:
                datasets.append(ds)
                sample_weights.append(np.full(len(ds), w / n_frames, dtype=np.float64))

        if bar is not None:
            bar.close()

        super().__init__(datasets)
        self.sample_weights = np.concatenate(sample_weights)

        # Persist any newly built entries so the next run is fast.
        if use_cache and fingerprint is not None and new_entries:
            merged = dict(index_cache or {})
            merged.update(new_entries)
            _save_index_cache(fingerprint, merged)
            print(f"[MixedSessionDataset] Saved {split} index cache "
                  f"({len(merged)} sessions) to {_INDEX_CACHE_DIR}")

    def make_sampler(self, num_samples: Optional[int] = None) -> torch.utils.data.WeightedRandomSampler:
        if num_samples is None:
            num_samples = len(self)
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(self.sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
        )


def _worker_init_fn(worker_id: int) -> None:
    """Pin per-worker thread pools so decode parallelism comes from the *process* count,
    not thread oversubscription.

    Each DataLoader worker otherwise inherits OpenCV's default thread count
    (cv2.getNumThreads() == #cores), so N workers spawn N x #cores decode threads fighting
    over #cores -- which slows JPEG decode, the dominant per-sample cost on these NFS-backed
    multi-view sessions. One cv2/torch thread per worker + many workers is the right shape.
    """
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


# ------------------------------------------------------------------- datamodule
class SessionDataModule(pl.LightningDataModule):
    """LightningDataModule for the multi-session WiLoR-v2 dataloader."""

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[MixedSessionDataset] = None
        self.val_dataset: Optional[MixedSessionDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = MixedSessionDataset(self.cfg, train=True)
            self.val_dataset = MixedSessionDataset(self.cfg, train=False)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        sampler = self.train_dataset.make_sampler()
        num_workers = self.cfg.GENERAL.NUM_WORKERS
        kwargs = {}
        if num_workers > 0:  # prefetch_factor / persistent_workers need workers
            kwargs["prefetch_factor"] = self.cfg.GENERAL.get("PREFETCH_FACTOR", 2)
            kwargs["persistent_workers"] = True
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=session_collate,
            pin_memory=True,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            **kwargs,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        num_workers = self.cfg.GENERAL.NUM_WORKERS
        kwargs = {}
        if num_workers > 0:  # prefetch_factor / persistent_workers need workers
            kwargs["prefetch_factor"] = self.cfg.GENERAL.get("PREFETCH_FACTOR", 2)
            kwargs["persistent_workers"] = True
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=session_collate,
            pin_memory=True,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            **kwargs,
        )

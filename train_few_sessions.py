"""Multi-view WiLoR-v2 training runner.

Orchestrates the pieces that already exist in this package:
    SessionDataModule  (datasets/mixed_session_dataset.py)
    MultiViewWiLoR     (models/multiview_wilor/multiview_wilor.py)
into a PyTorch-Lightning ``Trainer.fit`` with rf-detr-style wandb logging (scalars + lr + media).

The CLI mirrors rf-detr/train.py and exposes the same class of knobs WiLoR's default_config() /
pretrained model_config.yaml expose, so runs can be tweaked without editing yaml.

IMPORTANT: run from the prometheus repo root (``load_wilor`` resolves MANO files with repo-relative
paths). The train.sh wrapper cd's there for you; the smoke command in MULTIVIEW_MODEL.md uses
``python -m src.metric_hand_tracking_v2.wilor_v2.train``.
"""
import argparse
import re
from pathlib import Path
import inspect 
import warnings 

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import numpy as np
import torch

from src.metric_hand_tracking.wilor.configs import default_config, get_config
from src.metric_hand_tracking_v2.wilor_v2.datasets import SessionDataModule
from src.metric_hand_tracking_v2.wilor_v2.datasets.mixed_session_dataset import (
    MixedSessionDataset,
    discover_sessions,
)
from src.metric_hand_tracking_v2.wilor_v2.datasets.session_dataset import SessionDataset
from src.metric_hand_tracking_v2.wilor_v2.models.multiview_wilor import MultiViewWiLoR

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = "src/metric_hand_tracking/wilor/pretrained_models/wilor_final.ckpt"
_DEFAULT_WILOR_CFG = "src/metric_hand_tracking/wilor/pretrained_models/model_config.yaml"
_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "train_mix_sessions.yaml")


class _FewSessionDataset(MixedSessionDataset):
    """MixedSessionDataset but capped to `max_sessions` sessions per dataset entry.

    Slices the session list *before* any per-session init work, so startup is fast
    even when a dataset root contains hundreds of sessions on NFS.
    """

    def __init__(self, cfg, train: bool = True, max_sessions: int = 4):
        entries = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        weights = np.array([float(e["WEIGHT"]) for e in entries], dtype=np.float64)
        if abs(weights.sum() - 1.0) > 1e-6:
            raise ValueError(
                f"Dataset WEIGHTs must sum to 1.0 (got {weights.sum():.6f})"
            )

        datasets = []
        sample_weights_list = []
        for entry, w in zip(entries, weights):
            all_sessions = discover_sessions(entry["PATH"])
            sessions = all_sessions[:max_sessions]
            print(f"[FewSessionDataset] {entry['PATH']}: using {len(sessions)}/{len(all_sessions)} sessions")
            entry_datasets = [SessionDataset(cfg, p, train=train) for p in sessions]
            n_frames = sum(len(ds) for ds in entry_datasets)
            if n_frames == 0:
                raise FileNotFoundError(f"Dataset {entry['PATH']} has 0 frames")
            for ds in entry_datasets:
                datasets.append(ds)
                sample_weights_list.append(np.full(len(ds), w / n_frames, dtype=np.float64))

        torch.utils.data.ConcatDataset.__init__(self, datasets)
        self.sample_weights = np.concatenate(sample_weights_list)


class _FewSessionDataModule(SessionDataModule):
    def __init__(self, cfg, max_sessions: int = 4):
        super().__init__(cfg)
        self._max_sessions = max_sessions

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.train_dataset = _FewSessionDataset(self.cfg, train=True, max_sessions=self._max_sessions)
            self.val_dataset = _FewSessionDataset(self.cfg, train=False, max_sessions=self._max_sessions)


def _next_run_dir(base: Path, run_name: str) -> Path:
    """Return base/{slug}-{N} for the smallest N where the directory doesn't yet exist."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("_") or "run"
    i = 0
    while True:
        candidate = base / f"{slug}-{i}"
        if not candidate.exists():
            return candidate
        i += 1


def build_cfg(args: argparse.Namespace):
    """WiLoR default_config() + IMAGE_SIZE/aug fixes for hands + the clean data/train yaml +
    CLI overrides."""
    cfg = default_config()
    # SessionDataset reads MODEL.IMAGE_MEAN/IMAGE_STD/BBOX_SHAPE (and IMAGE_SIZE) for the crop
    # normalization; these live in the pretrained WiLoR model_config.yaml, not default_config().
    wilor_cfg = get_config(args.wilor_cfg, update_cachedir=True)
    cfg.MODEL.IMAGE_SIZE = wilor_cfg.MODEL.IMAGE_SIZE  # 256 for ViT WiLoR
    cfg.MODEL.IMAGE_MEAN = list(wilor_cfg.MODEL.IMAGE_MEAN)
    cfg.MODEL.IMAGE_STD = list(wilor_cfg.MODEL.IMAGE_STD)
    if "BBOX_SHAPE" in wilor_cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = list(wilor_cfg.MODEL.BBOX_SHAPE)
    # Required for 21-kp hands: extreme crop aug indexes body kp 25+ -> IndexError. Pinned off.
    cfg.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.0
    cfg.merge_from_file(args.config)

    # ---- CLI overrides onto the merged cfg (only when the flag was given) ----
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_EPOCHS = args.epochs
    cfg.GENERAL.NUM_WORKERS = args.num_workers
    if args.scale_factor is not None:
        cfg.DATASETS.CONFIG.SCALE_FACTOR = args.scale_factor
    if args.rot_factor is not None:
        cfg.DATASETS.CONFIG.ROT_FACTOR = args.rot_factor
    if args.rot_aug_rate is not None:
        cfg.DATASETS.CONFIG.ROT_AUG_RATE = args.rot_aug_rate
    if args.color_scale is not None:
        cfg.DATASETS.CONFIG.COLOR_SCALE = args.color_scale
    if args.do_flip is not None:
        cfg.DATASETS.CONFIG.DO_FLIP = args.do_flip
    cfg.freeze()
    return cfg


def build_logger(args: argparse.Namespace, output_dir: Path):
    if args.no_wandb:
        return None
    try:
        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(name=args.run_name, project=args.project, save_dir=str(output_dir))
    except ModuleNotFoundError as exc:  # graceful degradation, like rf-detr
        print(f"WandB logging disabled: {exc}. Install with: pip install wandb")
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train multi-view WiLoR-v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data / run
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="yacs data/train config")
    p.add_argument("--ckpt", type=str, default=_DEFAULT_CKPT, help="pretrained WiLoR checkpoint")
    p.add_argument("--wilor-cfg", type=str, default=_DEFAULT_WILOR_CFG, help="WiLoR model_config.yaml")
    p.add_argument("--output-dir", type=Path, default=Path("./experiments"))
    p.add_argument("--run-name", type=str, default="wilor-v2")
    # optimization (mirror WiLoR TRAIN.*)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-val", type=float, default=0.0, help="0 = off (manual clip in model)")
    # few-session mode
    p.add_argument("--max-sessions", type=int, default=4,
                   help="Max sessions to load per dataset entry (keeps startup fast)")
    # multi-view
    p.add_argument("--max-views", type=int, default=4)
    p.add_argument("--fusion-layers", type=int, default=8,
                   help="Number of alternating global/frame fusion layers (init from the ViT's last blocks)")
    p.add_argument("--fuse-camera-extrinsics", action="store_true",
                   help="Inject per-view camera pose (relative-to-reference) into the multi-view "
                        "fusion attention")
    # augmentation (mirror DATASETS.CONFIG.*); None -> keep config default
    p.add_argument("--scale-factor", type=float, default=None)
    p.add_argument("--rot-factor", type=float, default=None)
    p.add_argument("--rot-aug-rate", type=float, default=None)
    p.add_argument("--color-scale", type=float, default=None)
    flip = p.add_mutually_exclusive_group()
    flip.add_argument("--do-flip", dest="do_flip", action="store_true", default=None)
    flip.add_argument("--no-flip", dest="do_flip", action="store_false", default=None)
    # infra
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    # Debug caps (default 0/1.0 = no cap). Useful for quick wiring checks.
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = no cap (full epochs)")
    p.add_argument("--limit-train-batches", type=float, default=1.0)
    p.add_argument("--limit-val-batches", type=float, default=1.0)
    # logging
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--project", type=str, default="hand-tracking")
    p.add_argument("--log-every-n-steps", type=int, default=50)
    p.add_argument("--log-media-every-n-steps", type=int, default=500, help="0 = no media logging")
    p.add_argument("--num-log-images", type=int, default=4)
    args = p.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _next_run_dir(output_dir, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    cfg = build_cfg(args)
    dm = _FewSessionDataModule(cfg, max_sessions=args.max_sessions)
    model = MultiViewWiLoR(
        wilor_ckpt=args.ckpt,
        wilor_cfg=args.wilor_cfg,
        max_views=args.max_views,
        fusion_layers=args.fusion_layers,
        fuse_camera_extrinsics=args.fuse_camera_extrinsics,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_val=args.grad_clip_val,
        log_media_every_n_steps=args.log_media_every_n_steps,
        num_log_images=args.num_log_images,
    )

    logger = build_logger(args, output_dir)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir), monitor="val/loss", save_last=True, save_top_k=3
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator="auto",
        precision=args.precision,
        logger=logger if logger is not None else True,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        max_steps=args.max_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        # Manual optimization in the model -> Trainer auto-clip is forbidden; clip via --grad-clip-val.
        gradient_clip_val=None,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

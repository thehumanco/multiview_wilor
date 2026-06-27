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
import os
import re
from pathlib import Path
import warnings

# Must run before smplx/chumpy are imported (they get pulled in transitively when
# a MANO model is unpickled). Patches inspect.getargspec and the removed numpy
# scalar aliases that chumpy 0.70 still expects.
# import src.metric_hand_tracking.chumpy_compat  # noqa: F401

import inspect 
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
warnings.filterwarnings("ignore", category=FutureWarning)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.metric_hand_tracking.wilor.configs import default_config, get_config
from src.metric_hand_tracking_v2.wilor_v2.datasets import SessionDataModule
from src.metric_hand_tracking_v2.wilor_v2.datasets.mixed_session_dataset import _INDEX_CACHE_DIR
from src.metric_hand_tracking_v2.wilor_v2.models.multiview_wilor import MultiViewWiLoR

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = "src/metric_hand_tracking/wilor/pretrained_models/wilor_final.ckpt"
_DEFAULT_WILOR_CFG = "src/metric_hand_tracking/wilor/pretrained_models/model_config.yaml"
_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "train_mix_sessions.yaml")


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

        # entity must be explicit when the wandb account has no default entity set
        # server-side (otherwise init fails with "entityName required for model query").
        # Fall back to the logged-in user so a fresh account without a default
        # entity still works without passing --wandb-entity.
        # Precedence: explicit flag > $WANDB_ENTITY > the 'qasim_ali-gsi' team.
        # An explicit entity is required because the account's server-side default
        # entity resolves to a bogus "models" value (otherwise init fails with
        # "entityName required for model query"). The 'thehumanco' org currently
        # rejects writes ("user does not have models write access for this org"),
        # so default to the personal team that works.
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or "qasim_ali-gsi"
        return WandbLogger(
            name=args.run_name, project=args.project, save_dir=str(output_dir), entity=entity
        )
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
    p.add_argument("--output-dir", type=Path, default=Path("./src/metric_hand_tracking_v2/wilor_v2/experiments"))
    p.add_argument("--run-name", type=str, default="wilor-v2")
    # optimization (mirror WiLoR TRAIN.*)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=8.0e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-val", type=float, default=0.0, help="0 = off (manual clip in model)")
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
    p.add_argument("--num-workers", type=int, default=24,
                   help="DataLoader worker processes. Data-loading bound on these NFS multi-view "
                        "sessions, so scale toward the core count (cv2/torch are pinned to 1 "
                        "thread per worker in the dataloader's worker_init_fn).")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    # Debug caps (default 0/1.0 = no cap). Useful for quick wiring checks.
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = no cap (full epochs)")
    p.add_argument("--limit-train-batches", type=float, default=1.0)
    p.add_argument("--limit-val-batches", type=float, default=1.0)
    # logging
    p.add_argument("--delete-cache", action="store_true",
                   help=f"Delete the dataset index cache ({_INDEX_CACHE_DIR}) and rebuild from scratch")
    p.add_argument("--resume", type=str, default=None, help="Path to a .ckpt to resume training from (optimizer state, step, epoch are all restored)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="wandb entity (team/username). Defaults to the 'qasim_ali-gsi' team. "
                        "Override with this flag or $WANDB_ENTITY to log elsewhere.")
    p.add_argument("--project", type=str, default="multiview-wilor")
    p.add_argument("--log-every-n-steps", type=int, default=1000)
    p.add_argument("--log-media-every-n-steps", type=int, default=1000, help="0 = no media logging")
    p.add_argument("--num-log-images", type=int, default=4)
    args = p.parse_args()

    if args.delete_cache:
        if _INDEX_CACHE_DIR.is_dir():
            removed = list(_INDEX_CACHE_DIR.glob("*.json"))
            for f in removed:
                f.unlink()
            print(f"Deleted {len(removed)} cache file(s) from {_INDEX_CACHE_DIR}")
        else:
            print(f"No cache directory found at {_INDEX_CACHE_DIR}, nothing to delete")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _next_run_dir(output_dir, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    cfg = build_cfg(args)
    dm = SessionDataModule(cfg)
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
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()

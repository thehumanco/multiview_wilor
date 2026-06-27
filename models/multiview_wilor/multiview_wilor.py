"""MultiViewWiLoR: the multi-view hand-tracking main model.

Consumes the multi-session dataloader format (``session_collate`` output) and feeds the
single-view WiLoR regressor. Per frame it samples ``k ~ Uniform{1..MAX_VIEWS}`` available
views, concatenates all chosen crops into ONE batched WiLoR forward pass, then groups the
outputs back into per-(frame, view) dicts and hands them to ``compute_multiview_loss``
(which returns one loss per selected view).

Design (see MULTIVIEW_MODEL.md): one shared-weight WiLoR, a single batched forward over all
1-MAX_VIEWS selected-view crops (no padding, no per-view loop, no 4 separate weight sets).
"""
from typing import Dict, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.metric_hand_tracking.wilor.models import load_wilor

from .losses import compute_multiview_loss, discriminator_step
from .multiview_fusion import MultiViewFusion
from .view_sampling import select_views, ViewSelection

# GT MANO param keys carried per view from the collated batch (sliced by the view's row mask).
_GT_MANO_KEYS = ("global_orient", "hand_pose", "betas")


class MultiViewWiLoR(pl.LightningModule):
    def __init__(
        self,
        wilor_ckpt: str,
        wilor_cfg: str,
        max_views: int = 4,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        init_renderer: bool = False,
        grad_clip_val: float = 0.0,
        log_media_every_n_steps: int = 0,
        num_log_images: int = 4,
        fusion_layers: int = 8,
        fuse_camera_extrinsics: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # One shared WiLoR. load_wilor merges the pretrained model_config and restores weights.
        self.wilor, self.wilor_cfg = load_wilor(wilor_ckpt, wilor_cfg, init_renderer=init_renderer)
        if not init_renderer:
            self.wilor.renderer = None
            self.wilor.mesh_renderer = None
        self.wilor.automatic_optimization = True  # we drive optimization at this level

        # Multi-view fusion (VGGT-style alternating attention) between the frozen ViT trunk
        # and the per-view decoders. Identity at init (zero-gated), so step 0 == pretrained
        # WiLoR. Trainable: fusion + token decode heads + RefineNet; frozen: ViT trunk.
        # Optionally inject per-view camera extrinsics into the fusion attention (see
        # CameraExtrinsicsEmbed); it lives at the fusion entry.
        self._fuse_cam = fuse_camera_extrinsics
        self.fusion = MultiViewFusion.from_backbone(
            self.wilor.backbone, num_layers=fusion_layers,
            fuse_camera_extrinsics=fuse_camera_extrinsics,
        )
        bb = self.wilor.backbone
        bb.requires_grad_(False)
        for head in (bb.decpose, bb.decshape, bb.deccam):  # per-view decoders stay trainable
            head.requires_grad_(True)

        self.max_views = max_views
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_size = self.wilor_cfg.MODEL.IMAGE_SIZE
        self.grad_clip_val = grad_clip_val
        self.log_media_every_n_steps = log_media_every_n_steps
        self.num_log_images = num_log_images
        # ImageNet normalization used by WiLoR's crops; needed to un-normalize for media logging.
        self._img_mean = torch.tensor(self.wilor_cfg.MODEL.IMAGE_MEAN).view(3, 1, 1)
        self._img_std = torch.tensor(self.wilor_cfg.MODEL.IMAGE_STD).view(3, 1, 1)

        # Discriminator (adversarial training), same as WiLoR — only instantiated when the
        # ADVERSARIAL loss weight is > 0 so the default (0.0005 in pretrained config) enables it.
        self._use_disc = float(self.wilor_cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0.0)) > 0
        if self._use_disc or True:
            from src.metric_hand_tracking.wilor.models.discriminator import Discriminator
            self.discriminator = Discriminator()
        else:
            self.discriminator = None

        # Manual optimization (mirrors WiLoR, including the discriminator).
        self.automatic_optimization = False

    def train(self, mode: bool = True):
        """Keep the frozen ViT trunk in eval: its blocks carry heavy stochastic depth
        (drop_path up to 0.55) which would randomize the 'frozen' features."""
        super().train(mode)
        self.wilor.backbone.eval()
        return self

    def _generator_parameters(self) -> List[torch.nn.Parameter]:
        """All trainable params optimized by the generator step (everything but the disc)."""
        return [
            p for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith("discriminator")
        ]

    # --------------------------------------------------------------------- losses
    @property
    def _loss_kwargs(self) -> Dict:
        """Keyword args for compute_multiview_loss — bundles the shared loss modules + weights."""
        return dict(
            kp2d_loss=self.wilor.keypoint_2d_loss,
            kp3d_loss=self.wilor.keypoint_3d_loss,
            param_loss=self.wilor.mano_parameter_loss,
            loss_weights=self.wilor_cfg.LOSS_WEIGHTS,
            discriminator=self.discriminator,
        )

    # --------------------------------------------------------------------- helpers
    def _gather_crops(self, batch: Dict[int, Dict], sel: List[ViewSelection]) -> torch.Tensor:
        """Concat the chosen crops across all selected (frame, view) pairs into one tensor."""
        imgs = [batch[s.view]["img"][s.row_mask] for s in sel]
        return torch.cat(imgs, dim=0)

    def _predicted_K(self, focal_length: torch.Tensor) -> torch.Tensor:
        """Pinhole intrinsics in the crop frame: focal in px, principal point at crop center."""
        n = focal_length.shape[0]
        K = torch.zeros(n, 3, 3, device=focal_length.device, dtype=focal_length.dtype)
        c = self.image_size / 2.0
        K[:, 0, 0] = focal_length[:, 0]
        K[:, 1, 1] = focal_length[:, 1]
        K[:, 0, 2] = c
        K[:, 1, 2] = c
        K[:, 2, 2] = 1.0
        return K

    def _gather_flat_gt(self, batch: Dict[int, Dict], sel: List[ViewSelection]) -> Dict:
        """Gather every GT field across all selected (frame, view) rows into flat tensors in
        the SAME row order as ``_gather_crops``, moved to the device ONCE per field.

        ``_split_outputs`` then slices these on-device tensors per view, instead of issuing a
        separate host->device copy per field per view (dozens of tiny synchronous transfers a
        step). ``non_blocking=True`` lets the copies overlap compute since the dataloader pins
        memory (``pin_memory=True``).
        """
        def cat(field: str) -> torch.Tensor:
            return torch.cat(
                [batch[s.view][field][s.row_mask] for s in sel], dim=0
            ).to(self.device, non_blocking=True)

        flat = {
            "extrinsics": cat("extrinsics"),
            "R_crop_correction": cat("R_crop_correction"),
            "hand_id": cat("hand_id"),
            "keypoints_3d": cat("keypoints_3d"),
            "keypoints_2d": cat("keypoints_2d"),
            "right": cat("right"),
            "mano_params": {
                k: torch.cat(
                    [batch[s.view]["mano_params"][k][s.row_mask] for s in sel], dim=0
                ).to(self.device, non_blocking=True)
                for k in _GT_MANO_KEYS
            },
        }
        # per-hand MANO-GT mask (0 for keypoint-only hands, e.g. egoexo4d). Absent on older
        # batches -> None, and the loss falls back to all-ones.
        flat["has_mano"] = cat("has_mano") if "has_mano" in batch[sel[0].view] else None
        return flat

    def _split_outputs(
        self, out: Dict, imgs: torch.Tensor, flat_gt: Dict, sel: List[ViewSelection]
    ) -> List[Dict]:
        """Scatter the single batched WiLoR output back into one dict per selected view,
        attaching that view's GT. Every value is an on-device slice of ``out`` (predictions),
        ``imgs`` (crops, in the same row order), or ``flat_gt`` (GT gathered once by
        ``_gather_flat_gt``) — no per-field host->device copies. ``out``/``imgs``/``flat_gt`` are
        all in ``sel`` row order with the same per-selection offsets, so ``rows`` indexes them
        identically. See losses.compute_multiview_loss for the contract."""
        K = self._predicted_K(out["focal_length"])
        has_mano = flat_gt["has_mano"]
        per_view: List[Dict] = []
        offset = 0
        for s in sel:
            n = s.num_hands
            rows = slice(offset, offset + n)
            offset += n

            per_view.append({
                "frame": s.frame,
                "view": s.view,
                "num_hands": n,
                # --- predictions ---
                "pred_mano_params": {k: v[rows] for k, v in out["pred_mano_params"].items()},
                "pred_cam": out["pred_cam"][rows],
                "pred_cam_t": out["pred_cam_t"][rows],
                "pred_keypoints_3d": out["pred_keypoints_3d"][rows],
                "pred_keypoints_2d": out["pred_keypoints_2d"][rows],
                "focal_length": out["focal_length"][rows],
                "K": K[rows],
                "extrinsics": flat_gt["extrinsics"][rows],
                # crop->camera rotation (undoes the crop tilt); needed to map predictions to world
                "R_crop_correction": flat_gt["R_crop_correction"][rows],
                # per-frame hand identity; matches the same physical hand across views
                "hand_id": flat_gt["hand_id"][rows],
                # --- ground truth (masked to this view's valid hands) ---
                "gt_mano_params": {k: v[rows] for k, v in flat_gt["mano_params"].items()},
                "gt_keypoints_3d": flat_gt["keypoints_3d"][rows],
                "gt_keypoints_2d": flat_gt["keypoints_2d"][rows],
                "gt_has_mano": has_mano[rows] if has_mano is not None else None,
                "right": flat_gt["right"][rows],
                # Normalized crop pixels, kept only for media logging (not used by the loss).
                "img": imgs[rows],
            })
        return per_view

    def _fuse_views(
        self, tokens: torch.Tensor, batch: Dict[int, Dict], sel: List[ViewSelection],
        extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run multi-view fusion over the flat token batch, returning fused tokens in the
        same row order.

        Rows are grouped by (frame, hand_id) — the views of one physical hand — then hands
        are bucketed by view count V so each fusion call sees a rectangular (G, V, N, C)
        tensor (attention is length-agnostic; bucketing avoids padded tokens entirely).

        ``extrinsics`` (when fusing camera pose) is the on-device ``(total_hands, 3, 4)`` tensor
        already gathered by ``_gather_flat_gt`` in the same flat row order as ``tokens`` — reused
        here so we don't re-gather + re-copy it from the CPU batch.
        """
        # flat row index -> (frame, hand_id) group. hand_id lives on the CPU batch, so .tolist()
        # is a host-only read (no GPU sync) — keep the grouping on the CPU side.
        groups: Dict[tuple, List[int]] = {}
        offset = 0
        for s in sel:
            hids = batch[s.view]["hand_id"][s.row_mask].tolist()
            for j, hid in enumerate(hids):
                groups.setdefault((s.frame, int(hid)), []).append(offset + j)
            offset += s.num_hands

        # bucket hand-groups by view count (one rectangular fusion call per bucket), laid out
        # contiguously in ONE permutation so tokens/extrinsics are gathered with a single
        # host->device index transfer instead of one per bucket.
        by_v: Dict[int, List[List[int]]] = {}
        for rows in groups.values():
            by_v.setdefault(len(rows), []).append(rows)

        order: List[int] = []
        spans: List[tuple] = []  # (V, G_v, start, end) slices into `order`
        for V, row_lists in by_v.items():
            start = len(order)
            for rows in row_lists:
                order.extend(rows)
            spans.append((V, len(row_lists), start, len(order)))
        perm = torch.tensor(order, device=tokens.device, dtype=torch.long)

        N, C = tokens.shape[1], tokens.shape[2]
        tokens_perm = tokens[perm]  # (M, N, C), gathered once
        extr_perm = (
            extrinsics[perm].to(tokens.dtype)
            if (self._fuse_cam and extrinsics is not None) else None
        )

        out_parts = []
        for V, G_v, start, end in spans:
            extr_b = extr_perm[start:end].view(G_v, V, 3, 4) if extr_perm is not None else None
            fused = self.fusion(tokens_perm[start:end].view(G_v, V, N, C), extr_b)
            out_parts.append(fused.reshape(G_v * V, N, C))

        # invert the gather permutation (perm is a bijection over the M rows): scatter arange to
        # build the inverse, then gather back to original flat order — cheaper than argsort.
        fused_perm = torch.cat(out_parts, dim=0)  # in `perm` order
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return fused_perm[inv]

    # --------------------------------------------------------------------- forward
    def forward_step(self, batch: Dict[int, Dict]) -> Dict:
        sel = select_views(batch, max_views=self.max_views, min_views=2, train=self.training)
        if len(sel) == 0:
            return {"selections": [], "per_view": []}

        imgs = self._gather_crops(batch, sel).to(self.device, non_blocking=True)
        # Frozen trunk -> alternating-attention fusion across each hand's views -> the
        # pretrained decode heads + RefineNet (per view, unchanged).
        with torch.no_grad():
            tokens, (Hp, Wp) = self.wilor.backbone.forward_tokens(imgs[:, :, :, 32:-32])
        # Gather GT before fusion: the non_blocking H2D copies overlap the fusion compute, and
        # _fuse_views reuses the on-device extrinsics instead of re-copying them from the batch.
        flat_gt = self._gather_flat_gt(batch, sel)
        fused = self._fuse_views(tokens, batch, sel, flat_gt["extrinsics"])
        backbone_out = self.wilor.backbone.decode_tokens(fused, Hp, Wp)
        out = self.wilor.forward_step({"img": imgs}, train=self.training, backbone_out=backbone_out)
        per_view = self._split_outputs(out, imgs, flat_gt, sel)
        return {"selections": sel, "per_view": per_view}

    def forward(self, batch: Dict[int, Dict]) -> Dict:
        return self.forward_step(batch)

    # --------------------------------------------------------------------- logging
    def _log_stats(self, per_view: List[Dict], mode: str) -> None:
        """Log loss-agnostic batch statistics (view/hand counts) we own regardless of the loss.

        ``train/*`` go on both step and epoch; ``val/*`` only on epoch (mirrors rf-detr)."""
        on_step = mode == "train"
        num_views = len(per_view)
        num_hands = sum(int(pv["num_hands"]) for pv in per_view)
        avg = num_hands / max(num_views, 1)
        bs = max(num_hands, 1)
        self.log(f"{mode}/num_views", float(num_views), on_step=on_step, on_epoch=True, batch_size=bs)
        self.log(f"{mode}/num_hands", float(num_hands), on_step=on_step, on_epoch=True, batch_size=bs)
        self.log(f"{mode}/avg_hands_per_view", avg, on_step=on_step, on_epoch=True, batch_size=bs)

    def _log_breakdown(self, breakdown: Dict[str, torch.Tensor], mode: str, bs: int) -> None:
        """Log per-term loss scalars (per-view means + cross-view consistency terms)."""
        on_step = mode == "train"
        for name, val in breakdown.items():
            self.log(f"{mode}/{name}", val, on_step=on_step, on_epoch=True, batch_size=bs)

    @pl.utilities.rank_zero.rank_zero_only
    def _log_predictions(self, per_view: List[Dict], mode: str) -> None:
        """One wandb image per frame: columns = views, rows = hands within that view.

        Each cell shows the crop with GT keypoints (lime) and predicted keypoints (red).
        Views with fewer hands get blank cells so the grid is rectangular. Up to
        ``self.num_log_images`` frames are logged.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        mean = self._img_mean.cpu().numpy().reshape(3, 1, 1)
        std  = self._img_std.cpu().numpy().reshape(3, 1, 1)
        s = float(self.image_size)

        # Group per-view dicts by frame index.
        frames: Dict[int, List[Dict]] = {}
        for pv in per_view:
            frames.setdefault(pv["frame"], []).append(pv)

        wandb_images = []
        for frame_idx, views in list(frames.items())[:self.num_log_images]:
            views = sorted(views, key=lambda v: v["view"])
            num_cols = len(views)
            num_rows = max(pv["num_hands"] for pv in views)  # tallest column

            cell = s / 96.0  # inches per cell at 96 dpi → each crop renders at ~s px
            fig, axes = plt.subplots(
                num_rows, num_cols,
                figsize=(cell * num_cols, cell * num_rows),
                squeeze=False,
            )
            fig.subplots_adjust(wspace=0.02, hspace=0.02)

            for col, pv in enumerate(views):
                for row in range(num_rows):
                    ax = axes[row][col]
                    ax.axis("off")

                    if row == 0:
                        ax.set_title(f"v{pv['view']}", fontsize=6, pad=2)

                    if row >= pv["num_hands"]:
                        # blank cell — view has fewer hands than the tallest column
                        ax.set_facecolor("#111111")
                        continue

                    # un-normalize crop. .float() guards against bf16/16-mixed autocast outputs,
                    # which numpy() cannot convert.
                    img_t = pv["img"][row].detach().float().cpu().numpy()   # (3,H,W)
                    img = (img_t * std + mean).clip(0, 1).transpose(1, 2, 0)  # (H,W,3)

                    # keypoints: [-0.5, 0.5] → pixel coords
                    pred = pv["pred_keypoints_2d"][row].detach().float().cpu().numpy() * s + s / 2.0
                    gt_raw = pv["gt_keypoints_2d"][row].detach().float().cpu().numpy()
                    gt_px  = gt_raw[:, :2] * s + s / 2.0

                    ax.imshow(img, interpolation="bilinear")
                    # GT: hollow lime rings (visible even when pred overlaps exactly)
                    ax.scatter(gt_px[:, 0], gt_px[:, 1],
                               s=40, linewidths=1.0, zorder=3,
                               facecolors="none", edgecolors="lime")
                    # Pred: filled red dots drawn on top
                    ax.scatter(pred[:, 0], pred[:, 1],
                               s=12, linewidths=0, zorder=4,
                               c="red")

            wandb_images.append(wandb.Image(fig, caption=f"frame {frame_idx}"))
            plt.close(fig)

        if wandb_images:
            self.logger.experiment.log(
                {f"{mode}/predictions": wandb_images, "global_step": self.global_step}
            )

    def _should_log_media(self) -> bool:
        return (
            self.log_media_every_n_steps > 0
            and self.global_step > 0
            and self.global_step % self.log_media_every_n_steps == 0
        )

    # --------------------------------------------------------------------- steps
    def training_step(self, batch: Dict[int, Dict], batch_idx: int) -> Dict:
        optimizers = self.optimizers(use_pl_optimizer=True)
        opt_g = optimizers[0] if self._use_disc else optimizers
        output = self.forward_step(batch)
        per_view = output["per_view"]
        if len(per_view) == 0:
            return output  # nothing valid this batch; skip

        # ---- generator step (WiLoR + adversarial generator loss) ----------------
        losses, breakdown = compute_multiview_loss(per_view, **self._loss_kwargs)
        loss = torch.stack(list(losses)).sum()

        opt_g.zero_grad()
        self.manual_backward(loss)
        if self.grad_clip_val > 0:
            gn = torch.nn.utils.clip_grad_norm_(self._generator_parameters(), self.grad_clip_val)
            self.log("train/grad_norm", gn, on_step=True, on_epoch=False)
        opt_g.step()

        # ---- discriminator step (real GT vs fake predictions) -------------------
        if self._use_disc:
            opt_d = optimizers[1]
            loss_disc = discriminator_step(
                self.discriminator, per_view,
                loss_weight=float(self.wilor_cfg.LOSS_WEIGHTS["ADVERSARIAL"]),
                optimizer=opt_d,
                backward_fn=self.manual_backward,
            )
            self.log("train/loss_disc", loss_disc, on_step=True, on_epoch=False)

        output["losses"] = losses
        output["loss"] = loss.detach()
        nh = max(sum(int(pv["num_hands"]) for pv in per_view), 1)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=nh)
        self._log_breakdown(breakdown, "train", nh)
        self._log_stats(per_view, "train")
        if self._should_log_media() or (self.log_media_every_n_steps > 0 and batch_idx == 0):
            self._log_predictions(per_view, "train")
        return output

    def validation_step(self, batch: Dict[int, Dict], batch_idx: int) -> Dict:
        output = self.forward_step(batch)
        per_view = output["per_view"]
        if len(per_view) == 0:
            return output
        losses, breakdown = compute_multiview_loss(per_view, **self._loss_kwargs)
        loss = torch.stack(list(losses)).sum()
        output["losses"] = losses
        output["loss"] = loss.detach()
        nh = max(sum(int(pv["num_hands"]) for pv in per_view), 1)
        self.log("val/loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=nh)
        self._log_breakdown(breakdown, "val", nh)
        self._log_stats(per_view, "val")
        # Log media from the first validation batch each epoch.
        if self.log_media_every_n_steps > 0 and batch_idx == 0:
            self._log_predictions(per_view, "val")
        return output

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self._generator_parameters(),  # wilor (trainable parts) + fusion, excl. discriminator
            lr=self.lr, weight_decay=self.weight_decay,
        )
        if not self._use_disc:
            return opt_g
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        return [opt_g, opt_d]

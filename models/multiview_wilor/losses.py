"""Per-view supervised losses for multi-view WiLoR-v2.

Mirrors ``WiLoR.compute_loss`` (src/metric_hand_tracking/wilor/models/wilor.py) but operates
on the per-(frame, view) output dicts produced by ``MultiViewWiLoR.forward_step``.

CONTRACT — each element of ``per_view_outputs`` is a dict for ONE selected (frame, view),
holding ``num_hands`` (>=1) hands. All tensors are batched along dim 0 == num_hands.

Predictions (from WiLoR.forward_step, already grouped to this view):
    pred_mano_params: {
        'global_orient': (N, 1, 3, 3)   # root rotation R, rotmat
        'hand_pose':     (N, 15, 3, 3)  # theta, rotmats
        'betas':         (N, 10)
    }
    pred_cam:           (N, 3)          # [s, tx, ty] weak-perspective
    pred_cam_t:         (N, 3)          # full-perspective translation
    pred_keypoints_3d:  (N, 21, 3)      # MANO joints, camera frame
    pred_keypoints_2d:  (N, 21, 2)      # projected into crop frame
    focal_length:       (N, 2)
    K:                  (N, 3, 3)       # predicted pinhole intrinsics (crop frame)
    extrinsics:         (N, 3, 4)       # GT world->cam [R|t] for this view (static calib)

Ground truth (from the dataloader, masked to this view's valid hands):
    gt_mano_params: {'global_orient': (N,3), 'hand_pose': (N,45), 'betas': (N,10)}  # axis-angle
    gt_keypoints_3d: (N, 21, 4)   # camera frame, +conf
    gt_keypoints_2d: (N, 21, 3)   # crop frame, +conf
    gt_betas:        (N, 10)
    right:           (N,)         # 1.0 right / 0.0 left
    img_size:        (N, 2)       # [W, H]
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.metric_hand_tracking.wilor.models.losses import (
    Keypoint2DLoss,
    Keypoint3DLoss,
    ParameterLoss,
)
from src.metric_hand_tracking.wilor.utils.geometry import aa_to_rotmat

# Cross-view consistency loss weight keys (all optional; absent/0 => term is skipped).
_CONSISTENCY_KEYS = (
    "CONSISTENCY_BETAS",
    "CONSISTENCY_HAND_POSE",
    "CONSISTENCY_GLOBAL_ORIENT",
    "CONSISTENCY_KP3D",
    "CONSISTENCY_KP3D_ABS",
)


def _euclidean_variance(x: torch.Tensor) -> torch.Tensor:
    """Mean over views of the squared deviation from the cross-view mean.

    x: (V, ..., D). Treats the last dim as the vector; sums squared error over it, then means
    over views and any middle dims (e.g. keypoints/joints). For V==2 this equals half the squared
    pairwise difference. Differentiable, no reduction surprises.
    """
    mean = x.mean(dim=0, keepdim=True)
    return ((x - mean) ** 2).sum(dim=-1).mean()


def _rot_chordal_variance(R: torch.Tensor) -> torch.Tensor:
    """Mean over views of the squared Frobenius deviation from the cross-view mean rotation.

    R: (V, ..., 3, 3). Chordal distance is monotonic in geodesic angle, fully differentiable, and
    needs no SVD/log-map. The Euclidean mean is used as the centroid (a valid spread measure even
    though it is not itself a rotation).
    """
    mean = R.mean(dim=0, keepdim=True)
    return ((R - mean) ** 2).flatten(start_dim=-2).sum(dim=-1).mean()


def _hand_groups(per_view_outputs: List[Dict]):
    """Yield, per physical hand seen in >=2 selected views, its list of ``(out, row)`` rows.

    Groups by (frame, hand_id) exactly as the cross-view terms require: same physical hand,
    one row per view it is visible in. Singletons (seen in <2 views) are skipped.
    """
    frames: Dict[int, List[Dict]] = {}
    for out in per_view_outputs:
        frames.setdefault(out["frame"], []).append(out)
    for outs in frames.values():
        hand_obs: Dict[int, List[Tuple[Dict, int]]] = {}
        for out in outs:
            hids = out["hand_id"]
            for row in range(out["num_hands"]):
                hand_obs.setdefault(int(hids[row]), []).append((out, row))
        for obs in hand_obs.values():
            if len(obs) >= 2:
                yield obs


def compute_fused_world_kp3d_loss(
    per_view_outputs: List[Dict], kp3d_loss: Keypoint3DLoss, weight: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Supervise the cross-view-FUSED, world-frame 3D keypoints against GT.

    For each physical hand seen in >=2 views we map every view's predicted keypoints into the
    shared world frame (the same crop->camera->world chain ``W = R_w2c^T @ R_crop_correction``
    as ``compute_consistency_loss``), AVERAGE them into one fused estimate, and supervise it
    against the identically mapped GT with the standard root-relative, confidence-weighted L1
    (``kp3d_loss``). This is the train-time analog of the inference-time multi-view average:
    the gradient optimizes the fused estimate that is actually reported, rather than each view
    in isolation (the per-view KEYPOINTS_3D loss) or merely the cross-view spread
    (CONSISTENCY_KP3D). Root-relative -> depth-free.

    Note: the world map is an isometry, so for a SINGLE view this reduces to the per-view 3D
    loss; only the mean-over-views (taken in the common world frame, which is why W is needed)
    makes it distinct. Returns (weighted_loss, unweighted_detached) for logging.
    """
    device = per_view_outputs[0]["pred_mano_params"]["betas"].device
    dtype = per_view_outputs[0]["pred_mano_params"]["betas"].dtype
    z = torch.zeros((), device=device, dtype=dtype)
    if weight <= 0.0:
        return z, z  # off: true no-op

    pred_fused, gt_fused = [], []
    for obs in _hand_groups(per_view_outputs):
        pred = torch.stack([o["pred_keypoints_3d"][r] for o, r in obs])          # (V,21,3)
        gt = torch.stack([o["gt_keypoints_3d"][r] for o, r in obs])              # (V,21,4) +conf
        R_corr = torch.stack([o["R_crop_correction"][r] for o, r in obs])        # (V,3,3)
        ext = torch.stack([o["extrinsics"][r] for o, r in obs])                  # (V,3,4)
        W = ext[:, :, :3].transpose(-1, -2) @ R_corr                             # crop->world (V,3,3)

        # Left hands are predicted AND stored in a horizontally MIRRORED crop (WiLoR right-hand
        # convention). A mirror is not a rotation, so map it out (M = diag(-1,1,1)) on BOTH pred
        # and GT before the world mapping — see compute_consistency_loss for the full argument.
        if float(obs[0][0]["right"][obs[0][1]]) == 0.0:
            M = pred.new_tensor([-1.0, 1.0, 1.0])
            pred = pred * M
            gt = torch.cat([gt[..., :3] * M, gt[..., 3:]], dim=-1)

        pred_w = torch.einsum("vij,vkj->vki", W, pred - pred[:, :1, :])          # (V,21,3) world
        gt_w = torch.einsum("vij,vkj->vki", W, gt[..., :3] - gt[:, :1, :3])      # (V,21,3) world
        conf = gt[..., 3:].mean(dim=0)                                           # (21,1) shared validity

        pred_fused.append(pred_w.mean(dim=0))                                    # (21,3) fused pred
        gt_fused.append(torch.cat([gt_w.mean(dim=0), conf], dim=-1))            # (21,4) fused GT +conf

    if not pred_fused:
        return z, z
    loss = kp3d_loss(torch.stack(pred_fused), torch.stack(gt_fused), pelvis_id=0)
    return weight * loss, loss.detach()


def compute_consistency_loss(
    per_view_outputs: List[Dict], loss_weights: Dict
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Cross-view consistency: pull the predictions of the SAME physical hand toward agreement.

    For each hand seen in >=2 selected views (matched by (frame, hand_id)) we penalize the spread,
    across those views, of:
      - betas               (view-invariant shape)         -> CONSISTENCY_BETAS
      - hand_pose           (view-invariant articulation)  -> CONSISTENCY_HAND_POSE   [SO(3) chordal]
      - global_orient       mapped to WORLD frame          -> CONSISTENCY_GLOBAL_ORIENT [SO(3) chordal]
      - root-relative joints mapped to WORLD frame         -> CONSISTENCY_KP3D        [depth-free]
      - absolute joints      mapped to WORLD frame          -> CONSISTENCY_KP3D_ABS    [depth-polluted; default 0]

    World mapping uses W = R_w2c^T @ R_crop_correction (crop->camera->world); requires the crop
    correction to have been applied at train time. Returns (weighted_total, breakdown) where
    breakdown holds the unweighted per-term scalars for logging.
    """
    device = per_view_outputs[0]["pred_mano_params"]["betas"].device
    dtype = per_view_outputs[0]["pred_mano_params"]["betas"].dtype
    z = torch.zeros((), device=device, dtype=dtype)
    breakdown = {
        "loss_consistency_betas": z, "loss_consistency_hand_pose": z,
        "loss_consistency_global_orient": z, "loss_consistency_kp3d": z,
        "loss_consistency_kp3d_abs": z,
    }
    w = {k: float(loss_weights.get(k, 0.0)) for k in _CONSISTENCY_KEYS}
    if max(w.values()) <= 0.0:
        return z, breakdown  # all off: true no-op, keeps training identical to the GT-only baseline
    w_kpabs = w["CONSISTENCY_KP3D_ABS"]

    # Group selected views by frame, then collect each physical hand's rows across views.
    frames: Dict[int, List[Dict]] = {}
    for out in per_view_outputs:
        frames.setdefault(out["frame"], []).append(out)

    betas_t, hp_t, go_t, kp_t, kpabs_t = [], [], [], [], []
    for outs in frames.values():
        hand_obs: Dict[int, List[Tuple[Dict, int]]] = {}
        for out in outs:
            hids = out["hand_id"]
            for row in range(out["num_hands"]):
                hand_obs.setdefault(int(hids[row]), []).append((out, row))

        for obs in hand_obs.values():
            if len(obs) < 2:
                continue  # need >=2 views to have a spread
            betas = torch.stack([o["pred_mano_params"]["betas"][r] for o, r in obs])        # (V,10)
            hp = torch.stack([o["pred_mano_params"]["hand_pose"][r] for o, r in obs])        # (V,15,3,3)
            go = torch.stack([o["pred_mano_params"]["global_orient"][r, 0] for o, r in obs]) # (V,3,3)
            kp = torch.stack([o["pred_keypoints_3d"][r] for o, r in obs])                    # (V,21,3)
            cam_t = torch.stack([o["pred_cam_t"][r] for o, r in obs])                        # (V,3)
            # Left hands are predicted in a horizontally MIRRORED crop (WiLoR right-hand
            # convention), and a mirror is not a rotation: mapping it through the per-view
            # W below would make the cross-view spread unsatisfiable. Undo the x-flip
            # (M = diag(-1,1,1)) on everything that enters the world mapping. The true
            # orient is M @ go @ M, but the right factor (canonical-hand mirror) is shared
            # across views and drops out of the chordal spread, so left-multiply only.
            # betas/hand_pose spreads are invariant to the shared mirror -> untouched.
            o0, r0 = obs[0]
            if float(o0["right"][r0]) == 0.0:
                go = torch.cat([-go[:, :1, :], go[:, 1:, :]], dim=1)   # M @ go
                kp = kp * kp.new_tensor([-1.0, 1.0, 1.0])              # M @ kp
                cam_t = cam_t * cam_t.new_tensor([-1.0, 1.0, 1.0])
            R_corr = torch.stack([o["R_crop_correction"][r] for o, r in obs])               # (V,3,3)
            ext = torch.stack([o["extrinsics"][r] for o, r in obs])                          # (V,3,4)
            R_w2c, t_w2c = ext[:, :, :3], ext[:, :, 3]                                       # (V,3,3),(V,3)
            W = R_w2c.transpose(-1, -2) @ R_corr                                             # crop->world (V,3,3)

            betas_t.append(_euclidean_variance(betas))
            hp_t.append(_rot_chordal_variance(hp))
            go_t.append(_rot_chordal_variance(W @ go))                                       # world orient
            kp_rel = kp - kp[:, :1, :]                                                       # wrist-center
            kp_t.append(_euclidean_variance(torch.einsum("vij,vkj->vki", W, kp_rel)))        # root-rel world
            kps_world = torch.einsum("vij,vkj->vki", W, kp_rel)
            if w_kpabs > 0:
                X_cam = torch.einsum("vij,vkj->vki", R_corr, kp + cam_t[:, None, :])
                X_world = torch.einsum("vij,vkj->vki", R_w2c.transpose(-1, -2), X_cam - t_w2c[:, None, :])
                kpabs_t.append(_euclidean_variance(X_world))

    def _agg(terms: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(terms).mean() if terms else z

    L_betas, L_hp, L_go = _agg(betas_t), _agg(hp_t), _agg(go_t)
    L_kp, L_kpabs = _agg(kp_t), _agg(kpabs_t)
    breakdown = {
        "loss_consistency_betas": L_betas.detach(),
        "loss_consistency_hand_pose": L_hp.detach(),
        "loss_consistency_global_orient": L_go.detach(),
        "loss_consistency_kp3d": L_kp.detach(),
        "loss_consistency_kp3d_abs": L_kpabs.detach(),
    }
    total = (
        w["CONSISTENCY_BETAS"] * L_betas
        + w["CONSISTENCY_HAND_POSE"] * L_hp
        + w["CONSISTENCY_GLOBAL_ORIENT"] * L_go
        + w["CONSISTENCY_KP3D"] * L_kp
        + w_kpabs * L_kpabs
    )
    return total, breakdown


def compute_multiview_loss(
    per_view_outputs: List[Dict],
    kp2d_loss: Keypoint2DLoss,
    kp3d_loss: Keypoint3DLoss,
    param_loss: ParameterLoss,
    loss_weights: Dict,
    discriminator: Optional[nn.Module] = None,
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """Return ``(per_view_losses, breakdown)``.

    ``per_view_losses`` is a tuple of scalar losses summed by the caller: one per selected view,
    optionally an adversarial generator term, and (when any CONSISTENCY_* weight > 0) one cross-view
    consistency term. ``breakdown`` holds detached per-term scalars (per-view means + consistency
    terms) for logging.

    If ``discriminator`` is provided and ``loss_weights["ADVERSARIAL"] > 0``, a single
    adversarial generator loss is appended (it covers all views' predictions together, exactly as
    WiLoR does per batch).

    Args:
        per_view_outputs: list of per-(frame, view) output dicts; see module docstring.
        kp2d_loss:     Keypoint2DLoss instance (self.wilor.keypoint_2d_loss).
        kp3d_loss:     Keypoint3DLoss instance (self.wilor.keypoint_3d_loss).
        param_loss:    ParameterLoss instance (self.wilor.mano_parameter_loss).
        loss_weights:  dict-like with KEYPOINTS_2D/3D, GLOBAL_ORIENT, HAND_POSE, BETAS,
                       ADVERSARIAL keys (self.wilor_cfg.LOSS_WEIGHTS).
        discriminator: optional Discriminator module for adversarial generator loss.
    """
    if len(per_view_outputs) == 0:
        return (), {}

    losses = []
    term_acc = {"loss_2d": [], "loss_3d": [], "loss_go": [], "loss_hp": [], "loss_b": []}
    for out in per_view_outputs:
        n = out["num_hands"]
        device = out["pred_mano_params"]["betas"].device

        # ---- 2D keypoint loss ---------------------------------------------------
        # pred: (N, 21, 2)   gt: (N, 21, 3) with conf in last dim
        loss_2d = kp2d_loss(out["pred_keypoints_2d"], out["gt_keypoints_2d"])

        # ---- 3D keypoint loss (wrist-relative) ----------------------------------
        # pred: (N, 21, 3)   gt: (N, 21, 4) with conf in last dim
        loss_3d = kp3d_loss(out["pred_keypoints_3d"], out["gt_keypoints_3d"], pelvis_id=0)

        # ---- MANO parameter losses ----------------------------------------------
        pred_go = out["pred_mano_params"]["global_orient"]   # (N, 1, 3, 3)
        pred_hp = out["pred_mano_params"]["hand_pose"]       # (N, 15, 3, 3)
        pred_b  = out["pred_mano_params"]["betas"]           # (N, 10)

        gt_go_aa = out["gt_mano_params"]["global_orient"]    # (N, 3)  axis-angle
        gt_hp_aa = out["gt_mano_params"]["hand_pose"]        # (N, 45) axis-angle
        gt_b     = out["gt_mano_params"]["betas"]            # (N, 10)

        # Convert GT axis-angle → rotmat to match WiLoR's prediction space
        gt_go = aa_to_rotmat(gt_go_aa.reshape(-1, 3)).view(n, -1)   # (N, 9)
        gt_hp = aa_to_rotmat(gt_hp_aa.reshape(-1, 3)).view(n, -1)   # (N, 135)

        # Keypoint-only hands (e.g. egoexo4d) carry no MANO fit -> gt_has_mano masks the
        # MANO-parameter terms so the placeholder zeros don't supervise. Default: all GT present.
        has_all = out.get("gt_has_mano")
        has_all = (torch.ones(n, device=device, dtype=pred_b.dtype)
                   if has_all is None else has_all.to(device=device, dtype=pred_b.dtype))

        loss_go = param_loss(pred_go.reshape(n, -1), gt_go, has_all)
        loss_hp = param_loss(pred_hp.reshape(n, -1), gt_hp, has_all)
        loss_b  = param_loss(pred_b,                 gt_b,  has_all)

        # ---- weighted sum -------------------------------------------------------
        loss = (
            loss_weights["KEYPOINTS_2D"]    * loss_2d
            + loss_weights["KEYPOINTS_3D"]  * loss_3d
            + loss_weights["GLOBAL_ORIENT"] * loss_go
            + loss_weights["HAND_POSE"]     * loss_hp
            + loss_weights["BETAS"]         * loss_b
        )
        for k, v in zip(term_acc, (loss_2d, loss_3d, loss_go, loss_hp, loss_b)):
            term_acc[k].append(v.detach())
        losses.append(loss)

    # ---- adversarial generator loss (one term covering all views) ---------------
    # Collect predictions from every view into a single tensor and try to fool the
    # discriminator: we want disc_out → 1 (real), so loss = mean((disc_out - 1)^2).
    if discriminator is not None and loss_weights["ADVERSARIAL"] > 0:
        all_hp = torch.cat([o["pred_mano_params"]["hand_pose"] for o in per_view_outputs], dim=0)
        all_b  = torch.cat([o["pred_mano_params"]["betas"]     for o in per_view_outputs], dim=0)
        n_all  = all_hp.shape[0]
        disc_out = discriminator(all_hp.reshape(n_all, -1), all_b)
        loss_adv = ((disc_out - 1.0) ** 2).sum() / n_all
        losses.append(loss_weights["ADVERSARIAL"] * loss_adv)

    # ---- cross-view consistency (weighted; a no-op when all CONSISTENCY_* weights are 0) --------
    consistency_total, consistency_breakdown = compute_consistency_loss(per_view_outputs, loss_weights)
    if consistency_total.requires_grad or float(consistency_total) != 0.0:
        losses.append(consistency_total)

    # ---- fused world-frame 3D keypoint supervision (no-op when KEYPOINTS_3D_WORLD == 0) ---------
    w_kp3d_world = float(loss_weights.get("KEYPOINTS_3D_WORLD", 0.0))
    kp3d_world_total, kp3d_world_unw = compute_fused_world_kp3d_loss(
        per_view_outputs, kp3d_loss, w_kp3d_world
    )
    if kp3d_world_total.requires_grad or float(kp3d_world_total) != 0.0:
        losses.append(kp3d_world_total)

    breakdown = {k: torch.stack(v).mean() for k, v in term_acc.items() if v}
    breakdown.update(consistency_breakdown)
    breakdown["loss_consistency"] = consistency_total.detach()
    breakdown["loss_kp3d_world"] = kp3d_world_unw
    return tuple(losses), breakdown


def discriminator_step(
    discriminator: nn.Module,
    per_view_outputs: List[Dict],
    loss_weight: float,
    optimizer: torch.optim.Optimizer,
    backward_fn,
) -> torch.Tensor:
    """Train the discriminator for one step and return the detached loss.

    Uses GT hand poses from the batch as "real" samples and predicted poses (detached)
    as "fake" samples — the same least-squares GAN objective as WiLoR.

    Args:
        discriminator:    Discriminator module.
        per_view_outputs: per-(frame, view) output dicts (same as passed to compute_multiview_loss).
        loss_weight:      ADVERSARIAL weight (scales the disc loss before backprop).
        optimizer:        Discriminator optimizer.
        backward_fn:      ``self.manual_backward`` (Lightning's wrapper for AMP etc.).
    """
    # ---- collect real (GT) and fake (pred) poses across all views ---------------
    real_hp_list, real_b_list = [], []
    fake_hp_list, fake_b_list = [], []
    for out in per_view_outputs:
        n = out["num_hands"]
        device = out["pred_mano_params"]["betas"].device

        # real: convert GT axis-angle → rotmat. Skip keypoint-only hands (no MANO fit) so the
        # discriminator never sees placeholder-zero poses as "real".
        has_mano = out.get("gt_has_mano")
        keep = (slice(None) if has_mano is None
                else has_mano.to(device=device).bool())
        gt_hp_aa = out["gt_mano_params"]["hand_pose"][keep]    # (M, 45)
        gt_b     = out["gt_mano_params"]["betas"][keep]        # (M, 10)
        m = gt_b.shape[0]
        if m == 0:
            continue
        gt_hp_rm = aa_to_rotmat(gt_hp_aa.reshape(-1, 3)).view(m, -1)  # (M, 135)

        real_hp_list.append(gt_hp_rm)
        real_b_list.append(gt_b)

        # fake: detach predictions so disc gradients don't flow into WiLoR
        fake_hp_list.append(out["pred_mano_params"]["hand_pose"].reshape(n, -1).detach())
        fake_b_list.append(out["pred_mano_params"]["betas"].detach())

    if not real_hp_list:  # whole batch is keypoint-only -> no "real" MANO to train the disc on
        device = per_view_outputs[0]["pred_mano_params"]["betas"].device
        return torch.zeros((), device=device)
    real_hp = torch.cat(real_hp_list, dim=0)
    real_b  = torch.cat(real_b_list,  dim=0)
    fake_hp = torch.cat(fake_hp_list, dim=0)
    fake_b  = torch.cat(fake_b_list,  dim=0)
    n_total = real_hp.shape[0]

    disc_real = discriminator(real_hp, real_b)
    disc_fake = discriminator(fake_hp, fake_b)
    loss_real = ((disc_real - 1.0) ** 2).sum() / n_total
    loss_fake = ((disc_fake - 0.0) ** 2).sum() / n_total
    loss_disc = loss_weight * (loss_real + loss_fake)

    optimizer.zero_grad()
    backward_fn(loss_disc)
    optimizer.step()

    return loss_disc.detach()

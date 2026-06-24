"""End-to-end smoke test for MultiViewWiLoR on a few real frames.

Run from the prometheus REPO ROOT (load_wilor uses repo-relative MANO paths and we import
``src.``):

    conda activate prometheus
    OPENBLAS_NUM_THREADS=1 \
    python \
        -m src.metric_hand_tracking_v2.wilor_v2.models.multiview_wilor.test_smoke

Checks: dataloader -> collate -> MultiViewWiLoR.forward_step -> loss stub on 3-5 frames,
asserting per-view output counts match valid-hand counts and printing timing.
"""
import inspect
import os
import sys
import time

# Older chumpy/smplx pkl files (WiLoR's MANO) call inspect.getargspec, removed in Python 3.11+.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import torch
from yacs.config import CfgNode as CN

# Make `import src...` work from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.metric_hand_tracking.wilor.configs import default_config
from src.metric_hand_tracking_v2.wilor_v2.models.multiview_wilor import MultiViewWiLoR
from src.metric_hand_tracking_v2.wilor_v2.datasets import (
    SessionDataset,
    discover_sessions,
    session_collate,
)

# One real session root (oakink) discovered from the data config.
OAKINK_ROOT = "/lambda/nfs/hfm/qasim/hand_kp_dataset/oakink/sessions"
WILOR_CKPT = "src/metric_hand_tracking/wilor/pretrained_models/wilor_final.ckpt"
WILOR_CFG = "src/metric_hand_tracking/wilor/pretrained_models/model_config.yaml"
BATCH_SIZE = 4


def build_cfg() -> CN:
    cfg = default_config()
    cfg.defrost()
    cfg.MODEL.IMAGE_SIZE = 256          # ViT WiLoR
    cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
    cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.0  # MUST be 0 for hands (21 kp)
    cfg.DATASETS.JOINTS3D_IN_WORLD = False
    cfg.DATASETS.FILTER_EMPTY_FRAMES = True
    cfg.DATASETS.CACHE_FRAME_INDEX = True
    cfg.freeze()
    return cfg


def test_camera_extrinsics_embed():
    """Fast, checkpoint-free unit test for CameraExtrinsicsEmbed (identity-at-init guard)."""
    from src.metric_hand_tracking_v2.wilor_v2.models.multiview_wilor.multiview_fusion import (
        CameraExtrinsicsEmbed,
    )

    dim = 32
    embed = CameraExtrinsicsEmbed(dim).eval()
    for V in (1, 2, 4):  # V=1 (reference-only group) must not crash
        G = 3
        R = torch.linalg.qr(torch.randn(G, V, 3, 3))[0]   # valid rotations
        t = torch.randn(G, V, 3)
        ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)      # (G, V, 3, 4)
        out = embed(ext)
        assert out.shape == (G, V, dim), out.shape
        # zero-init final Linear => embedding is exactly zero at init => an exact no-op when
        # added to the token stream => MultiViewFusion stays identity-at-init.
        assert out.abs().max().item() == 0.0, "camera embedding must be zero at init"
    print("[smoke] CameraExtrinsicsEmbed zero-at-init + V=1/2/4 shapes OK")


def main():
    torch.manual_seed(0)
    test_camera_extrinsics_embed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_cfg()

    # --- one session, small dataloader ---
    session = discover_sessions(OAKINK_ROOT)[0]
    print(f"[smoke] session: {session}")
    ds = SessionDataset(cfg, session, train=True)
    print(f"[smoke] frames in session: {len(ds)}, num_views: {ds.num_views}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=session_collate
    )
    batch = next(iter(loader))
    assert isinstance(batch, dict), "collate should return {view_idx: {...}}"
    print(f"[smoke] batch views: {sorted(batch.keys())}")
    for v, d in batch.items():
        print(f"        view {v}: {d['img'].shape[0]} hands, valid={d['valid'].sum().item()}")

    # --- model (fuse_camera_extrinsics=True exercises the extrinsics-gather path in
    # _fuse_views end-to-end; identity-at-init means outputs still match pretrained WiLoR) ---
    model = MultiViewWiLoR(
        WILOR_CKPT, WILOR_CFG, max_views=4, fuse_camera_extrinsics=True
    ).to(device)
    model.train()

    # --- forward ---
    t0 = time.time()
    out = model.forward_step(batch)
    dt = time.time() - t0
    sel = out["selections"]
    per_view = out["per_view"]
    total_crops = sum(s.num_hands for s in sel)
    print(f"[smoke] selected {len(sel)} (frame,view) pairs, {total_crops} crops, "
          f"forward {dt*1000:.0f} ms on {device}")

    # per-frame view-count sanity: 1..max_views
    from collections import Counter
    by_frame = Counter(s.frame for s in sel)
    for f, n in by_frame.items():
        assert 1 <= n <= 4, f"frame {f} got {n} views (expected 1..4)"
    print(f"[smoke] views per frame: {dict(by_frame)}")

    # --- shape / count assertions ---
    for s, pv in zip(sel, per_view):
        n = s.num_hands
        assert pv["num_hands"] == n
        go = pv["pred_mano_params"]["global_orient"]
        hp = pv["pred_mano_params"]["hand_pose"]
        be = pv["pred_mano_params"]["betas"]
        assert go.shape[0] == n and go.shape[-2:] == (3, 3), go.shape
        assert hp.shape[0] == n and hp.shape[-2:] == (3, 3), hp.shape
        assert be.shape == (n, 10), be.shape
        assert pv["K"].shape == (n, 3, 3), pv["K"].shape
        assert pv["pred_cam_t"].shape == (n, 3)
        assert pv["gt_keypoints_3d"].shape == (n, 21, 4)
    print(f"[smoke] per-view shapes OK")

    # --- loss stub (call directly; training_step needs a Lightning Trainer for optimizers) ---
    from src.metric_hand_tracking_v2.wilor_v2.models.multiview_wilor import (
        compute_multiview_loss,
    )
    losses, _breakdown = compute_multiview_loss(per_view, **model._loss_kwargs)
    assert len(losses) >= 1, len(losses)
    loss = torch.stack(list(losses)).sum()
    loss.backward()  # verify the graph reaches the model (incl. the camera-embed MLP)
    # camera-embed MLP must be connected to the graph (grads exist; zero at init is expected)
    cam_params = list(model.fusion.cam_embed.parameters())
    assert any(p.grad is not None for p in cam_params), "camera embed got no gradient"
    print(f"[smoke] loss returned {len(losses)} terms "
          f"(sum={loss.item():.4f}); backward OK, camera-embed grads present")

    # --- warm forward timing (exclude CUDA warmup) ---
    model.eval()
    with torch.no_grad():
        batch2 = next(iter(loader))
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out2 = model.forward_step(batch2)
        if device == "cuda":
            torch.cuda.synchronize()
        warm = time.time() - t0
    crops2 = sum(s.num_hands for s in out2["selections"])
    print(f"[smoke] warm forward: {warm*1000:.0f} ms for {crops2} crops on {device}")

    print("[smoke] PASS")


if __name__ == "__main__":
    main()

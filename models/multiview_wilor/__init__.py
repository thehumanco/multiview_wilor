"""Multi-view WiLoR main model.

Bridges the multi-session dataloader (``wilor_v2.datasets.SessionDataModule`` /
``session_collate``) with the single-view WiLoR regressor
(``src.metric_hand_tracking.wilor.models.WiLoR``).

Per frame it samples a random number ``k ~ Uniform{1..MAX_VIEWS}`` of available views
(clamped to how many the frame actually has), runs all chosen crops through ONE shared
WiLoR in a single batched forward pass, and returns 1-MAX_VIEWS per-view output groups
(global_orient / hand_pose / betas as rotmats + camera matrices + GT) for the loss.
"""
from .multiview_wilor import MultiViewWiLoR
from .view_sampling import select_views
from .losses import compute_multiview_loss

__all__ = ["MultiViewWiLoR", "select_views", "compute_multiview_loss"]

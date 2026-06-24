"""Per-frame view sampling for the multi-view WiLoR model.

The dataloader's ``session_collate`` returns ``{view_idx: {field: (N_hands, ...),
'valid': (N,), 'frame_index': (N,)}}`` -- hands for a given view stacked across every
frame in the batch. Each frame can see a different set of views (oakink=3, ho3d=5,
interhand=73), so view selection is done **per source frame**:

  1. ``available_b`` = views with >=1 *valid* hand belonging to frame ``b``.
  2. Draw a random count ``k_b ~ Uniform{1..max_views}`` (train) / fixed ``max_views``
     (eval), then take ``n_b = min(k_b, |available_b|)`` views.
  3. Sample ``n_b`` views uniformly without replacement (train) / lowest indices (eval).

The result is a flat list of ``ViewSelection`` records, one per chosen (frame, view)
pair, each carrying a boolean row-mask into that view's stacked tensors selecting exactly
the valid hands of that frame. The model uses these to gather crops for one batched WiLoR
forward and to group the outputs back per (frame, view).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class ViewSelection:
    frame: int            # source-frame index within the batch (collate 'frame_index')
    view: int             # view id (key into the collated batch dict)
    row_mask: torch.Tensor  # (N_hands_in_view,) bool: this frame's valid hands for this view
    num_hands: int        # row_mask.sum(), >= 1


def _valid_rows_for_frame(view_dict: Dict, frame: int) -> torch.Tensor:
    """Bool mask over this view's stacked hands: valid AND belonging to `frame`."""
    return view_dict["valid"] & (view_dict["frame_index"] == frame)


def select_views(
    batch: Dict[int, Dict],
    max_views: int = 4,
    min_views: int = 2,
    train: bool = True,
    generator: Optional[torch.Generator] = None,
) -> List[ViewSelection]:
    """Sample min_views..min(k, available) views per frame; see module docstring.

    Frames with fewer than ``min_views`` valid views are skipped entirely so the
    model always sees at least two perspectives per training/eval step.

    Args:
        batch: ``session_collate`` output, ``{view_idx: {...}}``.
        max_views: upper cap on views per frame.
        min_views: minimum required views; frames below this threshold are dropped.
        train: random count + random view subset when True; deterministic when False.
        generator: optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Flat list of ``ViewSelection`` (possibly empty if no frame meets min_views).
    """
    if len(batch) == 0:
        return []

    # frame ids present anywhere in the batch
    frame_ids = torch.unique(
        torch.cat([v["frame_index"] for v in batch.values()])
    ).tolist()

    selections: List[ViewSelection] = []
    for b in frame_ids:
        # views that have >=1 valid hand for this frame, with their row-masks
        avail: List[tuple] = []
        for view_id, view_dict in batch.items():
            mask = _valid_rows_for_frame(view_dict, b)
            if bool(mask.any()):
                avail.append((view_id, mask))
        A = len(avail)
        if A < min_views:
            continue  # not enough views for this frame — skip it

        if train:
            k = int(torch.randint(min_views, max_views + 1, (1,), generator=generator).item())
            n = min(k, A)
            perm = torch.randperm(A, generator=generator)[:n].tolist()
            chosen = [avail[i] for i in perm]
        else:
            n = min(max_views, A)
            # deterministic: lowest view ids
            chosen = sorted(avail, key=lambda x: x[0])[:n]

        for view_id, mask in chosen:
            selections.append(
                ViewSelection(
                    frame=int(b),
                    view=int(view_id),
                    row_mask=mask,
                    num_hands=int(mask.sum().item()),
                )
            )

    return selections

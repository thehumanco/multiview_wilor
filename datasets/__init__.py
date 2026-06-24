"""Session dataloaders for multi-view WiLoR-v2 training.

These modules deliberately avoid the upstream HaMeR ``webdataset``/``braceexpand`` imports,
so this package ``__init__`` only re-exports the session loaders (safe to import in the
prometheus env).
"""
from .session_dataset import SessionDataset
from .mixed_session_dataset import (
    MixedSessionDataset,
    SessionDataModule,
    session_collate,
    discover_sessions,
)

__all__ = [
    "SessionDataset",
    "MixedSessionDataset",
    "SessionDataModule",
    "session_collate",
    "discover_sessions",
]

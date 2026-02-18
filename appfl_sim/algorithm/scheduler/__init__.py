from .base_scheduler import BaseScheduler

__all__ = ["BaseScheduler"]

try:
    from .sync_scheduler import SyncScheduler

    __all__.append("SyncScheduler")
except Exception:  # pragma: no cover
    SyncScheduler = None

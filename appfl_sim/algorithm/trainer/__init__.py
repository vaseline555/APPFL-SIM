from .base_trainer import BaseTrainer

# Optional APPFL trainers (heavy dependencies may be unavailable in sim-only envs)
try:
    from .vanilla_trainer import VanillaTrainer
except Exception:  # pragma: no cover
    VanillaTrainer = None

try:
    from .iiadmm_trainer import IIADMMTrainer
except Exception:  # pragma: no cover
    IIADMMTrainer = None

try:
    from .iceadmm_trainer import ICEADMMTrainer
except Exception:  # pragma: no cover
    ICEADMMTrainer = None

try:
    from .monai_trainer import MonaiTrainer
except Exception:  # pragma: no cover
    MonaiTrainer = None

try:
    from .fedsb_trainer import FedSBTrainer
except Exception:  # pragma: no cover
    FedSBTrainer = None

from .sim_vanilla_trainer import VanillaTrainer as SimVanillaTrainer

__all__ = [
    "BaseTrainer",
    "VanillaTrainer",
    "IIADMMTrainer",
    "ICEADMMTrainer",
    "MonaiTrainer",
    "FedSBTrainer",
    "SimVanillaTrainer",
]

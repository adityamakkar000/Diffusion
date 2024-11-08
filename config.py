from dataclasses import dataclass, field
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from Diffusion.model import TrainingConfig
from typing import Optional, List, Tuple, Union, Dict, Any

@dataclass
class ExperimentConfig:



    mode_type: Optional[str] = None
    diffusion_params: Optional[TrainingConfig] = None

    PATH: str = "model.pt"
    device: str = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"



cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)
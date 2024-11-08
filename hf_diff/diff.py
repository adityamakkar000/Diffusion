from dataclasses import dataclass
import huggingface_hub
from diffusers import UNet2DModel
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore


@dataclass
class HFTrainingConfig:
    image_size = 64  # the generated image resolution
    in_channels = 3  # number of input channels
    out_channels = 3
    layers_per_block = 2
    block_out_channels = (32, 64, 128)
    down_block_types = ("AttnDownBlock2D", "DownBlock2D", "DownBlock2D")
    up_block_types = ("UpBlock2D", "UpBlock2D", "AttnUpBlock2D")


cs = ConfigStore.instance()
cs.store(name="config", node=HFTrainingConfig)

def createHFDiffusion(config):
    return model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
    )


if __name__ == '__main__':

    print(OmegaConf.to_yaml(HFTrainingConfig))
from diffusers import UNet2DModel
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class UNet2DModelConfig:
    image_size: int = MISSING
    in_channels: int = MISSING
    out_channels: int = MISSING
    layers_per_block: int = MISSING
    block_out_channels: list = MISSING
    down_block_types: list = MISSING
    up_block_types: list = MISSING


def createHFDiffusion(config: UNet2DModelConfig) -> UNet2DModel:
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
    )

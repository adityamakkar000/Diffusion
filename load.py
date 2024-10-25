import torch
import torch.nn.functional as F
from torch import Tensor

from setupDataset import get_dataloaders
import matplotlib.pyplot as plt
from hf_diff.diff import model

import time
from itertools import cycle
import argparse

from config import ExperimentConfig

config = ExperimentConfig()

PATH = "diff_" + config.PATH
device = config.device

checkpoint = torch.load(PATH, weights_only=True, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print(
  "loaded model from checkpoint at step",
  checkpoint["step"],
  "with loss",
  checkpoint["loss"],
  )



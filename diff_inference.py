import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from setupDataset import get_dataloaders
import matplotlib.pyplot as plt
from hf_diff.diff import model

import argparse

from config import ExperimentConfig



config = ExperimentConfig()

batch_size_train = config.batch_size_train
batch_size_accumlation_multiple = config.batch_size_accumulation_multiple
batch_size_test = config.batch_size_test
lr = config.lr
max_steps = config.max_steps
scale = config.scale
size = config.size

B_1 = config.B_1
B_T = config.B_T
T = config.T

diffusion_params = config.diffusion_params

PATH = "diff_" + config.PATH

device = config.device

# linear based noise scheduler
beta_array = torch.linspace(B_1, B_T, T)
alpha_bar_array = torch.zeros(T).to(device)
alpha_bar_array[0] = 1 - beta_array[0]
for i in range(1, T):
    alpha_bar_array[i] = alpha_bar_array[i - 1] * (1 - beta_array[i])


model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load(PATH, weights_only=True, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"loaded model from step {checkpoint['step']} with loss {checkpoint['loss']}")
print("generating samples...")

with torch.no_grad():
    model.eval()

    x_T = torch.randn(1, 3, size[0], size[1]).to(device)
    img = model.inference(x_T, alpha_bar_array)

    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np + 1) / 2 * 255
    img_np = img_np.astype(np.uint8)

    plt.imsave('samples/generated_image.png', img_np)
    print(img)
    print("Image saved as 'generated_image.png'")


import torch
import torch.nn.functional as F
from torch import Tensor
from itertools import cycle

from setupDataset import get_dataloaders
import matplotlib.pyplot as plt
from DDPM.model import UNET

import time

batch_size_train = 128
batch_size_accumlation_multiple = 4
batch_size_test = 10
lr = 0.001

PATH = 'model.pt'

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


train_data, test_data = get_dataloaders(batch_size_train, batch_size_test, device)
train_data_iterator = iter(cycle(train_data)) 
test_data_iterator = iter(test_data)

# linear based noise scheduler

B_1 = 10 ** -4
B_T = 0.02
T = 1000
def create_random_scheduler(t: int) -> float:
  slope = (B_T - B_1)/(T - 1)
  return slope * (t - 1) + B_1


noise_arr = torch.zeros(T).to(device)
noise_arr[0] = 1 - create_random_scheduler(1)
for i in range(1,T):
  noise = create_random_scheduler(i + 1)
  noise_arr[i] = noise_arr[i-1] * (1 - noise)


def get_alpha(t: Tensor) -> float:
  noise = noise_arr[t].view(-1, 1,1,1)
  assert noise.dim() == 4 and noise.shape[0] == t.shape[0] and noise.shape[1] == noise.shape[2] == noise.shape[3]
  return noise

scale = 4
size = (218//scale,178//scale)

model = UNET(
        timeStep=T,
        orginalSize=size,
        inChannels=3,
        channels=[32,64,128],
        strides=[2,2],
        n_heads=[1],
        attn=[True, False, False],
        resNetBlocks=[2,2,2],
        dropout=[0.2]

    )

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)

print("model params", sum(p.numel() for p in model.parameters()))
print("starting training ...")

max_steps = 1000

for _ in range(max_steps):

    optimizer.zero_grad()
    loss_metric = 0
    start = time.time()
    for b in range(batch_size_accumlation_multiple):
      data, target = next(train_data_iterator)
      batch_image = data.to(device)
      batch_image = F.interpolate(batch_image, size)


      t = torch.randint(1, T, (batch_size_train,)).int().to(device)
      alpha = get_alpha(t)
      z = torch.randn_like(batch_image)
      single_image_noise = torch.sqrt(alpha) * batch_image + torch.sqrt(1 - alpha) * z
      single_image_noise = single_image_noise.to(device)

      predicted_noise = model(batch_image, t)

      loss = (1/( batch_size_accumlation_multiple)) * F.mse_loss(predicted_noise, z)
      loss.backward()
      loss_metric += loss.item()

    optimizer.step()

    if _ % 10 == 0:
      torch.save({
                'step': _,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_metric
                                 }, PATH)


    if device == 'cuda':
      torch.cuda.synchronize()
    end = time.time() - start
    batch_idx = ((_ + 1) * batch_size_accumlation_multiple) % len(train_data)
    percentage_complete = 100.0 * (_ + 1) / max_steps
    batch_percentage_complete = 100.0 * (batch_idx) / len(train_data)
    print(f'Step {_}/{max_steps} | Batch {batch_percentage_complete:.2f}% | Loss: {loss_metric:.6f} | Time: {end:.2f}s | {percentage_complete:.2f}% complete')

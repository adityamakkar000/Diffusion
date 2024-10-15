
import torch
from setupDataset import get_dataloaders

train_data, test_data = get_dataloaders()


import matplotlib.pyplot as plt

# cosine based noise schedulr
B_1 = 10 ** -4
B_T = 0.02
T = 1000
def create_random_scheduler(t: int) -> float:
  slope = (B_T - B_1)/(T - 1)
  return slope * (t - 1) + B_1


noise_arr = torch.zeros(1000)
noise_arr[0] = 1 - create_random_scheduler(1)
for i in range(1,1000):
  noise = create_random_scheduler(i + 1)
  noise_arr[i] = noise_arr[i-1] * (1 - noise)


for batch_idx, (example_data, example_target) in enumerate(train_data):

  ind = torch.randint(low=0, high=len(example_data), size=(1,))
  single_image = example_data[ind].squeeze(dim=0)

  plt.imshow( single_image.permute(1,2, 0) )
  t = torch.randint(low=1, high=1000, size=(1,))

  noise_current = noise_arr[t]
  z = torch.randn_like(single_image)
  single_image_noise = torch.sqrt(noise_current) * single_image + (1 - noise_current) * z
  plt.imshow(single_image_noise.permute(1,2,0))

  print(t)
  print(noise_current)


  break

import torch
import typing


# cosine based noise schedulr
def create_random_scheduler(t: int) -> float:
  slope = (0.02 - 10 ** (-4))/(999)
  y_int = 10 ** -4 - slope

  return slope * t + y_int

import torch
import torch.nn.functional as F
from torch import Tensor
from setupDataset import get_dataloaders
import matplotlib.pyplot as plt
from models.DDPM.model import UNET
from models.hf_diff.diff import createHFDiffusion
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig) -> None:

    if 'load' in cfg:
        load = True
        path = f"runs/{cfg.path}"
        assert os.path.exists(path), f"path, {path}, does not exist"
        cfg = OmegaConf.load(os.path.join(path, "config.yaml"))

    else:
       ValueError("load not in config")

    if not os.path.exists(f"{path}/samples"):
        os.makedirs(f"{path}/samples")

    print(OmegaConf.to_yaml(cfg))

    lr = cfg.training.lr
    size = cfg.training.size
    B_1 = cfg.training.B_1
    B_T = cfg.training.B_T
    T = cfg.training.T
    diffusion_params = cfg.model_config

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # linear based noise scheduler
    beta_array = torch.linspace(B_1, B_T, T)
    alpha_bar_array = torch.zeros(T).to(device)
    alpha_bar_array[0] = 1 - beta_array[0]
    for i in range(1, T):
        alpha_bar_array[i] = alpha_bar_array[i - 1] * (1 - beta_array[i])

    def save_image(x, path):
        img = (x + 1.0) * 255.0 / 2.0
        img = img.type(torch.uint8)
        plt.imsave(path, img.permute(1,2,0).cpu().detach().numpy(), format="png")


    if cfg.model == 'self':
        model = UNET(**diffusion_params).to(device)
    elif cfg.model == 'HF':
        model = createHFDiffusion(diffusion_params).to(device)
    else:
        raise NotImplementedError("model not implemented")

    optimizer = torch.optim.Adam(model.parameters(), lr)


    checkpoint = torch.load(f"{path}/model.pt", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        "loaded model from checkpoint at step",
        checkpoint["step"],
        "with loss",
        checkpoint["loss"],
    )

    print("Files loaded, setting up model ...\n\n")
    print("device", device)
    print("model params", sum(p.numel() for p in model.parameters()))
    print("starting inference ... \n\n")

    with torch.no_grad():
        model.eval()

        x_t = torch.randn(1, 3, size[0], size[1]).to(device) # intitally set to normal distrubtion

        for t in tqdm(range(999, 0, -1), desc="Generating images"):

            mean = model(x_t, torch.Tensor([t]).to(device).long())
            if cfg.model == 'HF':
                mean = mean['sample']

            alpha_bar = alpha_bar_array[t]
            alpha_bar_sub1 = alpha_bar_array[t - 1]
            alpha_current = alpha_bar / alpha_bar_sub1

            x_t = torch.sqrt(1 / alpha_bar) * (
            x_t
            - ((1 - alpha_current) / (torch.sqrt(1 - alpha_bar))) * mean
            )
    #            x_t = torch.clip(x_t, -1, 1)
            if t > 1:
                z = torch.randn_like(x_t)

                sigma  =  (1 - alpha_bar_sub1)/ (1 - alpha_bar) * torch.sqrt( 1- alpha_current )
                x_t = sigma * z

            if t % 100 == 0:
                save_image(x_t[0], f"{path}/samples/{t}.png")

        save_image(x_t[0], f"{path}/generated_image.png")
        print(f"saved image at {path}/generated_image.png")

if __name__ == "__main__":
    main()

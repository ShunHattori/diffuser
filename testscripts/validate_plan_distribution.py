import json
import pdb
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch

import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.guides.policies import Policy


class Parser(utils.Parser):
    dataset: str = "maze2d-umaze-v1"
    config: str = "config.maze2d"


# ---------------------------------- setup ----------------------------------#

args = Parser().parse_args("plan")
# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)
observation = env.reset()

# ---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

n_sample_path = 50

# ---------------------------------- inference ----------------------------------#
diffusion_chain = utils.colab.run_diffusion(
    diffusion,
    dataset,
    observation,
    # start=np.array([1, 1, 0, 0]),
    # goal=np.array([3, 3, 0, 0]),
    start=np.array([8.88987660e-01, 8.04300845e-01, 0, 0]),
    goal=np.array([3.10172081e00, 2.90496016e0, 0, 0]),
    n_samples=n_sample_path,
    device="cuda:0",
    horizon=128,
)

print(diffusion_chain.shape)  # (65, 5, 128, 4) = (n_diffusion_steps + 1, n_samples, horizon, observation_dim + action_dim)


# ---------------------------------- plotting ----------------------------------#

fig, ax = plt.subplots()
for i in range(n_sample_path):
    ax.plot(diffusion_chain[-1, i, :, 0], diffusion_chain[-1, i, :, 1], c="k", alpha=0.05)

print(f"Trajectory: {diffusion_chain[-1, 0, :, :]}")

ax.set_aspect("equal")
plt.show()


"""
python testscripts/validate_plan_distribution.py --diffusion_loadpath 'f:diffusion/2e6step_ds256_H128_T256'
"""

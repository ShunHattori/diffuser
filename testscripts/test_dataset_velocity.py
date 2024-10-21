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
dataset = env.get_dataset()

print(dataset["actions"])

print(f"action prop")
print(dataset["actions"].shape)
print(f"min : {dataset['actions'].min()}, max : {dataset['actions'].max()}")
print(f"mean : {dataset['actions'].mean()}, std : {dataset['actions'].std()}")

print(f"obs prop")
print(dataset["observations"].shape)
print(f"min : {dataset['observations'].min()}, max : {dataset['observations'].max()}")
print(f"mean : {dataset['observations'].mean()}, std : {dataset['observations'].std()}")


fig, ax = plt.subplots()
ax.set_aspect("equal")
fig.set_size_inches(10, 10)
plot = ax.scatter(dataset["observations"][:, 0], dataset["observations"][:, 1], c=dataset["actions"][:, 0], s=0.1, alpha=0.05)
fig.colorbar(plot)
plt.show()

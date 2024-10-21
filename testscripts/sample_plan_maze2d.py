import json
import pdb
from os.path import join

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


print(observation.shape)
print(observation[-1].shape)

horizons = [32, 128, 256]  # short medium long horizons
diffusion_chain = [
    utils.colab.run_diffusion(
        diffusion,
        dataset,
        observation,
        start=np.array([1, 1, 0, 0]),
        goal=np.array([3, 3, 0, 0]),
        n_samples=5,
        device="cuda:0",
        horizon=horizon,
    )
    for horizon in horizons
]

"""
FOR MAZE2D UMAZE
train each model
python scripts/train.py --config config.maze2d --dataset maze2d-umaze-v1 --horizon 32 --n_train_steps 2000000 --prefix diffusion/2e6step
python scripts/train.py --config config.maze2d --dataset maze2d-umaze-v1 --horizon 128 --n_train_steps 2000000 --prefix diffusion/2e6step
python scripts/train.py --config config.maze2d --dataset maze2d-umaze-v1 --horizon 256 --n_train_steps 2000000 --prefix diffusion/2e6step

inference
python testscripts/sample_plan_maze2d.py --horizon 32 --diffusion_loadpath 'f:diffusion/H32_T64'
python testscripts/sample_plan_maze2d.py --horizon 128 --diffusion_loadpath 'f:diffusion/H128_T64'
python testscripts/sample_plan_maze2d.py --horizon 128 --diffusion_loadpath 'f:diffusion/2e6step_H128_T64'
python testscripts/sample_plan_maze2d.py --horizon 128 --diffusion_loadpath 'f:diffusion/H128_T256'
python testscripts/sample_plan_maze2d.py --horizon 256 --diffusion_loadpath 'f:diffusion/H256_T64'

FOR MAZE2D UMAZE with large diffusion steps
python scripts/train.py --config config.maze2d --dataset maze2d-umaze-v1 --horizon 128 --n_diffusion_steps 256 --n_train_steps 2000000 --prefix diffusion/2e6step_ds256
"""

print(diffusion_chain[0].shape)  # n_diffusion_step , batch , horizon, state dim

# utils.colab.show_sample(renderer, diffusion_chain[-1], savebase="./media")
for i, horizon in enumerate(horizons):
    utils.colab.show_diffusion(renderer, diffusion_chain[i], savebase=f"./media_{args.dataset}_trainH{args.horizon}_DS{args.n_diffusion_steps}_inferH{horizon}")  # sample horizon
# policy = Policy(diffusion, dataset.normalizer)


# # ---------------------------------- main loop ----------------------------------#

# if args.conditional:
#     print("Resetting target")
#     env.set_target()

# ## set conditioning xy position to be the goal
# target = env._target
# cond = {
#     diffusion.horizon - 1: np.array([*target, 0, 0]),
# }

# ## observations for rendering
# rollout = [observation.copy()]

# cond[0] = observation
# action, sample = policy(cond, batch_size=args.batch_size, horizon=args.horizon)  # batch_size = 1
# print(sample.observations.shape)  # (1, 128, 4) -> batch, horizon, state dim

# # ---------------------------------- rendering ----------------------------------#
# renderer.composite(join(args.savepath, "sameple.png"), [sample.observations[0, :, :2]], ncol=1)

# print(sample.observations.shape)

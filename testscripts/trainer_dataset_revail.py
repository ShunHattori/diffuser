import collections
import importlib
import os
import pickle

import torch

import diffuser.utils as utils

# # Load the dataset
# with open("/home/shun-hat/diffuser/logs/maze2d-umaze-v1/diffusion/H512_T256/dataset_config.pkl", "rb") as f:
#     dataset = pickle.load(f)


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Parser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d"


args = Parser().parse_args("diffusion")


dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),  # 読み込みではなく保存先 (BACKUP)
    env=args.dataset,
    horizon=args.horizon,  # デフォルトではホライゾンの長さを継承する
    # horizon=64,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

dataset = dataset_config()

# print(dataset.observation_dim)


# shuffle=Falseとすると，make_indicesで順番に軌道から部分パスを取り出している様子がわかる (ここで長さも制御している)
dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True))

# for batch in dataloader:
#     print(batch)  # データサイズがdatasetのhorizonに一致(今はmaze2d-large-v1なので384 (0~383))

# 一部取り出して表示してみる．連結された軌道かどうかを確認

import matplotlib.pyplot as plt

N = 100
for i in range(N):
    batch = dataloader.__next__()
    trajectories = batch.trajectories
    trajectories = trajectories.squeeze(0)
    print(trajectories.shape)
    plt.plot(trajectories[:, 2], trajectories[:, 3])  # x, y
plt.show()

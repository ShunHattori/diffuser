import torch

import diffuser.utils as utils


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
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,  # デフォルトではホライゾンの長さを継承する
    # horizon=64,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    num_limit_episodes=10,
)

dataset = dataset_config()


dataloader = cycle(
    torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # def:32
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
)

for i in range(20):
    data = next(dataloader)
    print(data.conditions[0][0, :])
    # num_limit_episodesを10に設定し，ループが20回だと同じデータが２回ランダムな順番で出力される つまり制限に成功
    # "num_limit_episodes": 10, で設定可能


"""
python testscripts/sample_traj_from_limited_dataset.py --config config.maze2d_limited_dataset --dataset maze2d-umaze-v1 --horizon 128

Output:

pybullet build time: Nov 28 2023 23:51:11
[ utils/colab ] Warning: not importing colab dependencies
[ utils/setup ] Reading config: config.maze2d_limited_dataset:maze2d_umaze_v1
[ utils/setup ] Using overrides | config: config.maze2d_limited_dataset | dataset: maze2d_umaze_v1
[ utils/setup ] Found extras: ['--horizon', '128']
[ utils/setup ] Overriding config | horizon : 512 --> 128
[ utils/setup ] Setting exp_name to: diffusion/H128_T256
[ utils/setup ] Saved args to logs/maze2d-umaze-v1/diffusion/H128_T256/args.json
[ utils/config ] Imported diffuser.datasets:GoalDataset_limited_episodes

[utils/config ] Config: <class 'diffuser.datasets.sequence.GoalDataset_limited_episodes'>
    env: maze2d-umaze-v1
    horizon: 128
    max_path_length: 40000
    normalizer: LimitsNormalizer
    num_limit_episodes: 10
    preprocess_fns: ['maze2d_set_terminals']
    use_padding: False

[ utils/config ] Saved config to: logs/maze2d-umaze-v1/diffusion/H128_T256/dataset_config.pkl

load datafile: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 45.60it/s]
[ utils/preprocessing ] Segmented maze2d-umaze-v1 | 1565 paths | min length: 96 | max length: 3993
dataset["timeouts"].shape: (1000000,)
dataset["timeouts"].sum(): 12459
dataset["timeouts"].shape: (1000000,)
dataset["timeouts"].sum(): 1566
[ datasets/buffer ] Finalized replay buffer | 1566 episodes
fields.n_episodes=1566
fields.path_lengths=array([ 267, 1199,  388, ..., 1375,  276,  583])
horizon=128
self.indices=array([[   0,    0,  128],
       [   0,    1,  129],
       [   0,    2,  130],
       ...,
       [1565,  452,  580],
       [1565,  453,  581],
       [1565,  454,  582]])
self.indices.shape=(798664, 3)
fields=[ datasets/buffer ] Fields:
    actions: (1566, 40000, 2)
    infos/goal: (1566, 40000, 2)
    infos/qpos: (1566, 40000, 2)
    infos/qvel: (1566, 40000, 2)
    observations: (1566, 40000, 4)
    rewards: (1566, 40000, 1)
    terminals: (1566, 40000, 1)
    timeouts: (1566, 40000, 1)
    next_observations: (1566, 40000, 4)
    normed_observations: (1566, 40000, 4)
    normed_actions: (1566, 40000, 2)
LIMITED EPISODES: len(self.indices)=10
self.indices=array([[  11,  827,  955],
       [ 605,  546,  674],
       [ 869, 1374, 1502],
       [1399,  209,  337],
       [ 320, 1160, 1288],
       [ 369,   75,  203],
       [ 223,   82,  210],
       [ 704,  127,  255],
       [1061,  208,  336],
       [ 687, 1005, 1133]])
self.indices.shape=(10, 3)
tensor([ 0.8447,  0.1545, -0.0943, -0.6784])
tensor([-0.6344,  0.7252,  0.1523,  0.3030])
tensor([-0.7520,  0.2883,  0.3487, -0.7288])
tensor([ 0.2279,  0.8925, -0.5143, -0.3725])
tensor([ 0.8094, -0.0186,  0.0255,  0.5311])
tensor([-0.5530,  0.0347, -0.0076, -0.5060])
tensor([ 0.3191,  0.7833, -0.9504,  0.0119])
tensor([ 0.0700,  0.7871,  0.5766, -0.1241])
tensor([ 0.3034,  0.7922,  0.8107, -0.0442])
tensor([ 0.6314,  0.8155, -0.5771,  0.0452])
tensor([-0.7520,  0.2883,  0.3487, -0.7288])
tensor([ 0.8447,  0.1545, -0.0943, -0.6784])
tensor([ 0.2279,  0.8925, -0.5143, -0.3725])
tensor([ 0.0700,  0.7871,  0.5766, -0.1241])
tensor([ 0.3191,  0.7833, -0.9504,  0.0119])
tensor([ 0.6314,  0.8155, -0.5771,  0.0452])
tensor([-0.5530,  0.0347, -0.0076, -0.5060])
tensor([-0.6344,  0.7252,  0.1523,  0.3030])
tensor([ 0.8094, -0.0186,  0.0255,  0.5311])
tensor([ 0.3034,  0.7922,  0.8107, -0.0442])

"""

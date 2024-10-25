import socket

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
]


plan_args_to_watch = [
    ("prefix", ""),
    ##
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("value_horizon", "V"),
    ("discount", "d"),
    ("normalizer", ""),
    ("batch_size", "b"),
    ##
    ("conditional", "cond"),
]

base = {
    "diffusion": {
        "seed": 42,
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 256,
        "n_diffusion_steps": 256,
        "action_weight": 1,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 4, 8),
        "renderer": "utils.Maze2dRenderer",
        ## dataset
        "loader": "datasets.GoalDataset_limited_episodes",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": ["maze2d_set_terminals"],
        "clip_denoised": True,
        "use_padding": False,
        "max_path_length": 40000,
        "num_limit_episodes": 50000,
        ## serialization
        "logbase": "logs",
        "prefix": "diffusion/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 2000000,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 20000,
        "sample_freq": 20000,
        "n_saves": 50,
        "save_parallel": False,
        "n_reference": 50,
        "n_samples": 10,
        "bucket": None,
        "device": "cuda",
    },
    "plan": {
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": 256,
        "n_diffusion_steps": 256,
        "normalizer": "LimitsNormalizer",
        ## serialization
        "vis_freq": 10,
        "logbase": "logs",
        "prefix": "plans/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        ## loading
        "diffusion_loadpath": "f:diffusion/H{horizon}_T{n_diffusion_steps}",
        "diffusion_epoch": "latest",
    },
}

# ------------------------ overrides ------------------------#

"""
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
"""

maze2d_umaze_v1 = {
    "diffusion": {
        "horizon": 128,
        "n_diffusion_steps": 64,
    },
    "plan": {
        "horizon": 128,
        "n_diffusion_steps": 64,
    },
}

maze2d_large_v1 = {
    "diffusion": {
        "horizon": 384,
        "n_diffusion_steps": 256,
    },
    "plan": {
        "horizon": 384,
        "n_diffusion_steps": 256,
    },
}


# データセット制限検証用モデル学習コマンド
# H128_DS256_TS2000000とH128_DS64_TS2000000に対して検証したいが，，，DS伸ばしたら学習にかかる時間どれだけ伸びたっけ？
# python scripts/train.py --config config.maze2d_limited_dataset --dataset maze2d-umaze-v1 --horizon 128 --n_diffusion_steps 64 --n_train_steps 2000000 --prefix diffusion/2e6step_LIMIT_5e4_
# python scripts/train.py --config config.maze2d_limited_dataset --dataset maze2d-umaze-v1 --horizon 128 --n_diffusion_steps 256 --n_train_steps 2000000 --prefix diffusion/2e6step_LIMIT_5e4_

# データセット制限検証用モデル学習コマンド 保存先を確認すること！
# python testscripts/sample_plan_maze2d.py --horizon 128 --n_diffusion_steps 256 --diffusion_loadpath 'f:diffusion/2e6step_LIMIT_5e4__H128_T256' --prefix 'plans/release_DL5e4_TS2e6'

import collections
import os
import pdb
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import gym
import numpy as np


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

# -----------------------------------------------------------------------------#
# -------------------------------- general api --------------------------------#
# -----------------------------------------------------------------------------#


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def get_dataset(env):
    dataset = env.get_dataset()

    if "antmaze" in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset


def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)  # ここで長いパスとして1mを分割(ゴール位置ベース)

    """
    ここで不要なパスを消去すればいい．
    パスの接続性に注意
    見当違い. 与えられるパスはもっと長い. 1M状態からなる1566本のパス
    そこからサンプリングしているだけであって，断片的なパスを入れているわけではない．
    """

    # if "maze2d" in env.name:
    #     print("partially removing maze2d paths......")
    #     print(f'number of paths: {len(dataset["rewards"])}')
    #     dataset = partially_remove_maze2d_paths(dataset)
    #     print(f'number of paths: {len(dataset["rewards"])}')

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        for k in dataset:
            if "metadata" in k:
                continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if "maze2d" in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


# -----------------------------------------------------------------------------#
# -------------------------------- maze2d fixes -------------------------------#
# -----------------------------------------------------------------------------#


def partially_remove_maze2d_paths(dataset):
    """
    Filters out paths that pass near a specified point within a given radius.

    Args:
        dataset (dict): Original dataset containing fields like 'observations', 'timeouts', etc.
        point (tuple): The point (x, y) to filter paths around.
        radius (float): The radius within which paths will be filtered out.

    Returns:
        dict: A new dataset with paths passing near the point removed.
    """
    # CUSTOMIZED FOR MAZE2D-UMANZE-V1
    remove_coord = np.array([1.8, 3.0])
    remove_judge_radius = 0.3

    timeout_indices = np.where(dataset["timeouts"] == 1)[0]
    keep_indices = np.ones(len(next(iter(dataset.values()))), dtype=bool)  # Boolean mask to keep track of indices to retain

    # Iterate over all paths using timeout indices
    for i in range(len(timeout_indices) - 1):
        start_idx = timeout_indices[i]
        end_idx = timeout_indices[i + 1]

        # Extract the segment of the path
        path_segment = dataset["observations"][start_idx:end_idx]

        # Check if any point in the path segment is within the given radius of the specified point
        distances = np.linalg.norm(path_segment[:, :2] - remove_coord, axis=1)
        if np.any(distances < remove_judge_radius):
            # Mark the entire path segment for removal
            keep_indices[start_idx : end_idx + 1] = False

    # Filter the dataset using the boolean mask
    filtered_dataset = {key: value[keep_indices] for key, value in dataset.items()}

    return filtered_dataset


def process_maze2d_episode(episode):
    """
    adds in `next_observations` field to episode
    """
    assert "next_observations" not in episode
    length = len(episode["observations"])
    next_observations = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode

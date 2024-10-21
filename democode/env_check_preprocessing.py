import gym
import matplotlib.pyplot as plt
import numpy as np

from diffuser.datasets.d4rl import get_dataset, load_environment
from diffuser.datasets.preprocessing import maze2d_set_terminals


def test_dataset_attributes(env_name):
    env = load_environment(env_name)
    # fn = maze2d_set_terminals(env)
    # dataset = fn(get_dataset(env))
    dataset = get_dataset(env)

    print(f'number of paths: {len(dataset["rewards"])}')
    dataset = partially_remove_maze2d_paths(dataset)
    print(f'number of paths: {len(dataset["rewards"])}')

    print("Dataset Keys and Shapes:")
    for key, value in dataset.items():
        print(f"{key}: shape {value.shape}")

    timeout_indices = np.where(dataset["timeouts"] == 1)[0]
    print(f"num timeouts: {len(timeout_indices)}")

    num_render_path = 100
    for i in range(num_render_path):
        # visualize the path
        plt.scatter(
            dataset["observations"][timeout_indices[i] : timeout_indices[i + 1], 0],
            dataset["observations"][timeout_indices[i] : timeout_indices[i + 1], 1],
            alpha=0.6,
            s=2,
        )
        # visualize the start and goal
        # plt.scatter(dataset["observations"][timeout_indices[i], 0], dataset["observations"][timeout_indices[i], 1])
    plt.show()


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


if __name__ == "__main__":
    test_dataset_attributes("maze2d-umaze-v1")  # 例として maze2d 環境を使用


"""
(diffuser) shun-hat@shunhat-desktop:~/diffuser$ python democode/env_check.py
pybullet build time: Nov 28 2023 23:51:11
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5 to /home/shun-hat/.d4rl/datasets/maze2d-medium-sparse-v1.hdf5
load datafile: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 21.34it/s]
Dataset Keys and Shapes:
actions: shape (2000000, 2)
infos/goal: shape (2000000, 2)
infos/qpos: shape (2000000, 2)
infos/qvel: shape (2000000, 2)
observations: shape (2000000, 4)
rewards: shape (2000000,)
terminals: shape (2000000,)
timeouts: shape (2000000,)
"""

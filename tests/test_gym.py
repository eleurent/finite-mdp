from __future__ import division, print_function
import gym

import finite_mdp


def test_deterministic():
    env = gym.make('finite-mdp-v0')
    env.configure({
        "mode": "deterministic",
        "transition": [[1, 2],
                       [0, 3],
                       [2, 2],
                       [3, 3]],
        "reward": [[0, 1000],
                   [0, -1000],
                   [0, 0],
                   [0, 0]]
    })
    obs = env.reset()
    for i in range(3):
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        print(obs, action, reward, new_obs)
        obs = new_obs
    env.close()
    assert env.observation_space.contains(obs)
    assert reward is not None
    assert not done


def test_stochastic():
    env = gym.make('finite-mdp-v0')
    env.configure({
        "mode": "stochastic",
        "transition": [[[0.1, 0.7, 0.1, 0.1], [0, 0, 1, 0]],
                       [[0.7, 0.1, 0.1, 0.1], [0, 0, 0, 1]],
                       [[0, 0, 1, 0], [0, 0, 1, 0]],
                       [[0, 0, 0, 1], [0, 0, 0, 1]]],
        "reward": [[0, 1000],
                   [0, -1000],
                   [0, 0],
                   [0, 0]]
    })
    obs = env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        print(obs, action, reward, new_obs)
        obs = new_obs
    env.close()
    assert env.observation_space.contains(obs)
    assert reward is not None
    assert not done

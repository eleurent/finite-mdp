import numpy as np
import json
from finite_mdp.mdp import MDP, StochasticMDP
import pytest


def test_conversion():
    deterministic_config = {
        "mode": "deterministic",
        "transition": [[1, 2],
                       [0, 3],
                       [2, 2],
                       [3, 3]],
        "reward": [[0, 1000],
                   [0, -1000],
                   [0, 0],
                   [0, 0]]
    }
    mdp = StochasticMDP.from_deterministic(MDP.from_config(deterministic_config))
    assert np.all(np.sum(mdp.transition, axis=-1))

    assert mdp.state == 0
    mdp.step(0)
    assert mdp.state == 1
    mdp.step(1)
    assert mdp.state == 3


def test_random(tmpdir):
    # Build random MDP
    config = {
        "mode": "uniform",
        "num_states": 100,
        "num_actions": 5
    }
    mdp = MDP.from_config(config)
    mdp.step(action=np.random.choice(range(mdp.transition.shape[1])))

    assert np.all(np.sum(mdp.transition, axis=-1))

    # Write to file
    with open(tmpdir.join("mdp.json"), 'w') as f:
        json.dump(mdp.to_config(), f)

    print(tmpdir)
    assert len(tmpdir.listdir()) == 1


def test_garnet():
    config = {
        "mode": "garnet",
        "num_states": 23,
        "num_actions": 3,
        "num_transitions": 2,
        "reward_sparsity": 0.2
    }
    mdp = MDP.from_config(config)
    mdp.step(action=np.random.choice(range(mdp.transition.shape[1])))

    assert np.all(np.sum(mdp.transition, axis=-1))
    assert np.all(np.count_nonzero(mdp.transition, axis=2) <= config["num_transitions"])
    assert np.count_nonzero(mdp.reward) == pytest.approx(config["reward_sparsity"] * mdp.reward.size, 1)


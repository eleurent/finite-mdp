import numpy as np
import json
from finite_mdp.mdp import MDP, StochasticMDP


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
    assert mdp.state == 0
    mdp.step(0)
    assert mdp.state == 1
    mdp.step(1)
    assert mdp.state == 3


def test_random(tmpdir):
    # Build random MDP
    config = {
        "mode": "deterministic",
        "random": True,
        "transition": np.zeros((100, 5)),
        "reward": np.zeros((100, 5))
    }
    mdp = MDP.from_config(config)

    # Write to file
    with open(tmpdir.join("mdp.json"), 'w') as f:
        json.dump(mdp.to_config(), f)

    print(tmpdir)
    assert len(tmpdir.listdir()) == 1

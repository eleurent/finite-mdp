from __future__ import division, print_function
import gym

import finite_mdp


def test_finite_mdp():
    env = gym.make('finite-mdp-v0')

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


if __name__ == "__main__":
    test_finite_mdp()
import numpy as np
import gym

from tile_coding.tiles import TileCoding


class EpsilonGreedy:
    def __init__(self, values, epsilon, features):
        self.qvalues = values
        self.eps = epsilon
        self.features = features

    def sample(self, state):
        if np.random.uniform() < self.eps:
            return int(np.random.randint(self.qvalues.shape[1]))
        return np.argmax(self.qvalues[self.features(state), :].sum(axis=0))


nsteps = 1000
nepisodes = 350
gamma = 0.99
exploration = 0.05
learning_rate = 0.1

env = gym.make('MountainCar-v0')

env._max_episode_steps = nsteps

features = TileCoding(ntiling=10, memory_size=3000)

states_num = features.memory_size
actions_num = env.action_space.n
qvalues = np.zeros((states_num, actions_num))

policy = EpsilonGreedy(qvalues, exploration, features)

for ep in range(nepisodes):
    cumul_reward = 0.
    state = env.reset()
    action = policy.sample(state)

    for step in range(nsteps):
        env.render()

        next_state, reward, done, _ = env.step(action)
        next_action = policy.sample(next_state)

        # TD error
        tderror = reward - qvalues[features(state), action].sum()
        if not done:
            tderror += gamma * qvalues[features(next_state), next_action].sum()

        # TD update
        qvalues[features(state), action] += (learning_rate / features.ntiling) * tderror

        # Q-learning housekeeping
        state, action = next_state, next_action

        cumul_reward += reward
        if done:
            break

    print("Episode {0} finished after {1} steps, undiscounted return {2}".format(ep, step + 1, cumul_reward))

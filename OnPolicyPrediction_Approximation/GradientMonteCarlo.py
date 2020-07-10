from Environment.random_walk_1000_states import RandomWalk1000
import numpy as np
import collections
import matplotlib.pyplot as plt
import random


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list/np.sum(probability_list)


class LinearFunction:
    def __init__(self,d):
        self.dim = d
        self.weight = np.random.rand(self.dim + 1, 1)

    def __call__(self, x):
        x_ = np.ones([self.dim + 1, 1])
        x_[1][0] = x
        return np.dot(self.weight.transpose(), x_)[0][0]


class Agent:
    def __init__(self, env, dimension):
        self.env = env
        self.value_state = LinearFunction(dimension)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def MC_app(self, number_of_episodes, learning_rate, gamma=0.9):
        for _ in range(number_of_episodes):
            episode = []
            state = self.env.reset()
            action = self.select_action(state)
            episode.append([0, state, action])
            while True:
                new_state, reward, is_done, _ = self.env.step(action)
                action = self.select_action(state)
                state = new_state
                episode.append([reward, state, action])
                if is_done:
                    break
            # update g base on g = gamma * g + R_{t+1}
            g = 0
            for i in range(len(episode)-1, -1, -1):
                g = gamma*g + episode[i][0]
                s = episode[i][1]
                if s is None:
                    continue
                # s /= 1000.
                delta_value = np.array([[1], [s]])
                self.value_state.weight += learning_rate * (g - self.value_state(s)) * delta_value


if __name__ == '__main__':
    env = RandomWalk1000()
    agent = Agent(env, 1)
    agent.MC_app(1000, 2e-5)
    x = np.arange(0, 1001, 1.)
    y = np.arange(0, 1001, 1.)
    for i in range(x.size):
        y[i] = agent.value_state(x[i])
    plt.plot(x,y)
    plt.show()

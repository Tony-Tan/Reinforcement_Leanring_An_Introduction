import collections
import numpy as np
from Environment.random_walk_19_states import RandomWalk
import matplotlib.pyplot as plt
import random


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env):
        self.env = env
        self.policies = collections.defaultdict(constant_factory(2))
        self.value_of_state = collections.defaultdict(lambda: 0.0)

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def estimating(self, iteration_times, lambda_coe=1., alpha=0.1, gamma=0.9, epsilon=0.3):
        for iter_time in range(iteration_times):
            episode_record = collections.deque()
            current_state = self.env.reset()
            current_action = self.select_action(current_state)
            next_state, reward, is_done, _ = self.env.step(current_action)
            episode_record.append([current_state, current_action, reward])
            while not is_done:
                current_state = next_state
                current_action = self.select_action(current_state)
                next_state, reward, is_done, _ = self.env.step(current_action)
                episode_record.append([current_state, current_action, reward])
            while len(episode_record) != 0:
                current_state, current_action, reward = episode_record.popleft()
                g_n = [reward]
                gamma_iter = gamma
                reward_cumulative = reward
                for n in range(0, len(episode_record)):
                    reward_cumulative += gamma_iter * episode_record[n][2]
                    gamma_iter *= gamma
                    if n + 1 < len(episode_record):
                        state_n_t = episode_record[n + 1][0]
                        g_n.append(reward_cumulative + gamma_iter * self.value_of_state[state_n_t])
                    else:
                        g_n.append(reward_cumulative)
                lambda_coe_temp = lambda_coe
                g_t_lambda = 0
                for g in g_n:
                    g_t_lambda += (1 - lambda_coe) * lambda_coe_temp * g
                    lambda_coe_temp *= lambda_coe
                self.value_of_state[current_state] += alpha * (g_t_lambda - self.value_of_state[current_state])
                # update policies
                # value_of_action_list = []
                # for action_iter in range(self.env.action_space.n):
                #     value_of_action_list.append(self.value_of_state[current_state])
                # value_of_action_list = np.array(value_of_action_list)
                # optimal_action = np.random.choice(
                #     np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                # for action_iter in range(self.env.action_space.n):
                #     if action_iter == optimal_action:
                #         self.policies[current_state][
                #             action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
                #     else:
                #         self.policies[current_state][action_iter] = epsilon / self.env.action_space.n


if __name__ == '__main__':
    env = RandomWalk(19)
    agent = Agent(env)
    agent.estimating(10, 0.8, 0.1, 1)
    value_of_state = []
    for i_state in range(1, env.state_space.n - 1):
        value_of_state.append(agent.value_of_state[i_state])
    plt.plot(value_of_state)
    plt.show()

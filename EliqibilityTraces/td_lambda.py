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

    def estimating(self, iteration_times, lambda_coe=1., alpha=0.1, gamma=0.9):
        for iter_time in range(iteration_times):
            eligibility_trace = collections.defaultdict(lambda: 0.0)
            current_state = self.env.reset()
            current_action = self.select_action(current_state)
            next_state, reward, is_done, _ = self.env.step(current_action)
            while True:
                for k in self.value_of_state.keys():
                    eligibility_trace[k] = gamma * lambda_coe * eligibility_trace[k]
                eligibility_trace[current_state] += 1.
                if not is_done:
                    delta_value = reward + gamma * self.value_of_state[next_state] - \
                                  self.value_of_state[current_state]
                else:
                    delta_value = reward - self.value_of_state[current_state]
                for k in self.value_of_state.keys():
                    self.value_of_state[k] += alpha * delta_value * eligibility_trace[k]
                if is_done:
                    break
                current_state = next_state
                current_action = self.select_action(current_state)
                next_state, reward, is_done, _ = self.env.step(current_action)


if __name__ == '__main__':
    env = RandomWalk(19)
    alpha_list = [i / 20. for i in range(1, 21)]
    lambda_list = [1., 0.99, 0.975, 0.95, 0.9, 0.8, 0.4, 0]
    ground_truth = [(-1. + i / 9.) for i in range(0, 19)]
    ground_truth = np.array(ground_truth)
    average_times = 100
    for j in range(len(lambda_list)):
        mse = []
        for alpha_i in range(0, len(alpha_list)):
            print(alpha_i, alpha_list[alpha_i])
            mse_current = 0
            for aver in range(average_times):
                agent = Agent(env)
                agent.estimating(10, lambda_list[j], alpha_list[alpha_i], 0.9)
                value_of_state = []
                for i_state in range(1, env.state_space.n - 1):
                    value_of_state.append(agent.value_of_state[i_state])
                value_of_state = np.array(value_of_state)
                mse_current += np.sum((value_of_state - ground_truth[1:env.state_space.n - 1]) *
                                      (value_of_state - ground_truth[1:env.state_space.n - 1])) / 17.

            mse.append(np.sqrt(mse_current / float(average_times)))
        max_i = -1
        for mse_i in mse:
            if mse_i < 1.0:
                max_i += 1
            else:
                break

        plt.plot(alpha_list[0:max_i], mse[0:max_i], label='$\\lambda = $' + str(lambda_list[j]))
    plt.legend()
    plt.show()

    # env = RandomWalk(19)
    # agent = Agent(env)
    # agent.estimating(10, 1, 1, 0.9)
    # value_of_state = []
    # for i_state in range(1, env.state_space.n - 1):
    #     value_of_state.append(agent.value_of_state[i_state])
    # plt.plot(value_of_state)
    # plt.show()

import collections
import numpy as np
from randomwalk import RandomWalk
import matplotlib.pyplot as plt

def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list/np.sum(probability_list)


class Agent:
    def __init__(self, n):
        self.n = n
        self.policies = collections.defaultdict(constant_factory(2))
        self.value_of_state = collections.defaultdict(lambda: 0.5)

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def estimating(self, env, iteration_times, alpha=0.9, gamma=0.9):
        for _ in range(iteration_times):
            current_stat = env.reset()
            action = self.select_action(current_stat)
            # the doc of deque can be found: https://docs.python.org/3/library/collections.html#collections.deque
            n_queue = collections.deque()
            new_state, reward, is_done, _ = env.step(action)
            while True:
                n_queue.append([new_state, reward, is_done])
                if is_done:
                    while len(n_queue) != 0:
                        state_updated, _, _ = n_queue.popleft()
                        gamma_temp = 1
                        G = 0
                        for iter_n in n_queue:
                            G += gamma_temp * iter_n[1]
                            gamma_temp *= gamma
                        self.value_of_state[state_updated] += (alpha * (G - self.value_of_state[state_updated]))
                    break
                else:
                    if len(n_queue) == self.n + 1:
                        state_updated, _, _ = n_queue.popleft()
                        gamma_temp = 1
                        G = 0
                        for iter_n in n_queue:
                            G += gamma_temp * iter_n[1]
                            gamma_temp *= gamma
                        action_next = self.select_action(new_state)
                        new_state, reward, is_done, _ = env.step(action_next)
                        G += (reward*gamma_temp+self.value_of_state[new_state])
                        self.value_of_state[state_updated] += (alpha * (G - self.value_of_state[state_updated]))


if __name__ == '__main__':
    env = RandomWalk(19)
    agent = Agent(2)

    ground_truth = []
    rms_array = []
    for i in range(19):
        ground_truth.append(-1+i/19)
    alpha_array = [i/100. for i in range(1, 100)]
    for alpha_i in alpha_array:
        value_list_of_state = np.array([0.0 for _ in range(19)])
        for _ in range(100):
            agent.estimating(env, 10, alpha=alpha_i)
            for i in range(env.state_space.n):
                value_list_of_state[i] += (agent.value_of_state[i])
        value_list_of_state = value_list_of_state / 100

        rms = np.sum((np.array(value_list_of_state[1:-1]) - np.array(ground_truth[1:-1]))**2)/17
        rms_array.append(rms)
    plt.plot(alpha_array, np.array(rms_array))
    plt.show()


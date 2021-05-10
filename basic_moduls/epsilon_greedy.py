import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon_):
        self._epsilon = epsilon_

    def __call__(self, states_value_array, action_space_n):
        prob = np.zeros(action_space_n)
        optimal_action = \
            np.random.choice(np.flatnonzero(states_value_array == states_value_array.max()))
        for action_iter in range(action_space_n):
            if action_iter == optimal_action:
                prob[action_iter] = 1. - self._epsilon + self._epsilon / action_space_n
            else:
                prob[action_iter] = self._epsilon / action_space_n
        return prob

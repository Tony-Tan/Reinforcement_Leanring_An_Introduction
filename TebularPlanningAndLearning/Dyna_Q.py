from Environment.gride_world import GridWorld
import numpy as np
import collections
import matplotlib.pyplot as plt
import random

def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list/np.sum(probability_list)


class Agent:
    def __init__(self, env, n=5, epsilon=0.4, initial_value=0.0):
        self.env = env
        self.epsilon = epsilon
        self.value_state_action = collections.defaultdict(lambda: initial_value)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))
        self.model = collections.defaultdict(lambda: {})
        self.model_state_action_list = []
        self.n = n

    def select_action(self, state):
        sum = np.sum(self.policies[state])
        self.policies[state] = self.policies[state] / sum
        probability_distribution = self.policies[state]
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def dyna_q(self, number_of_episodes, alpha=0.1, gamma=1, epsilon=0.4):
        for _ in range(number_of_episodes):
            state = self.env.reset()
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, is_block = self.env.step(action)
                if is_block:
                    self.policies[state][action] = 0.0
                    continue

                # next state-action return
                # if not is_done:
                q_state_next = []
                for action_iter in range(self.env.action_space.n):
                    q_state_next.append(self.value_state_action[(new_state, action_iter)])
                q_state_next = max(q_state_next)
                q_state_current = self.value_state_action[(state, action)]
                self.value_state_action[(state, action)] = \
                    q_state_current + alpha * (reward + gamma * q_state_next - q_state_current)
                # add into model
                self.model[state][action] = [reward, new_state]
                if [state, action] not in self.model_state_action_list:
                    self.model_state_action_list.append([state,action])

                # planning
                for n_iter in range(self.n):
                    state_selected, action_selected = random.choice(self.model_state_action_list)
                    reward_in_model, new_state_in_model = self.model[state_selected][action_selected]
                    q_state_next_in_model = []
                    for action_iter in range(self.env.action_space.n):
                        q_state_next_in_model.append(self.value_state_action[(new_state_in_model, action_iter)])
                    q_state_next_in_model = max(q_state_next_in_model)
                    q_state_current_in_model = self.value_state_action[(state_selected, action_selected)]
                    self.value_state_action[(state_selected, action_selected)] = \
                        q_state_current_in_model + alpha * (reward_in_model + gamma * q_state_next_in_model -
                                                            q_state_current_in_model)
                if is_done:
                    break
                state = new_state

            # control epsilon-greedy
            for state in range(self.env.state_space.n):
                value_of_action_list = []
                possible_action_num = self.env.action_space.n
                for action_iter in range(possible_action_num):
                    if self.policies[state][action_iter] == 0:
                        possible_action_num -= 1
                        value_of_action_list.append(-10.0)
                        continue
                    value_of_action_list.append(self.value_state_action[(state, action_iter)])
                value_of_action_list = np.array(value_of_action_list)
                optimal_action = \
                    np.random.choice(np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                for action_iter in range(self.env.action_space.n):
                    if self.policies[state][action_iter] == 0:
                        continue
                    if action_iter == optimal_action:
                        self.policies[state][action_iter] = 1 - epsilon + epsilon/possible_action_num
                    else:
                        self.policies[state][action_iter] = epsilon/possible_action_num


if __name__ == '__main__':
    env = GridWorld(8, [5, 13, 21, 10, 18, 26, 34, 38, 46])
    agent = Agent(env)
    agent.dyna_q(10000, alpha=0.1, gamma=0.6)
    env.plot_grid_world(agent.policies)
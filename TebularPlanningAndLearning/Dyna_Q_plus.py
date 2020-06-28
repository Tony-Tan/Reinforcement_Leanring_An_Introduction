from Environment.gride_world import GridWorld
import numpy as np
import collections
import matplotlib.pyplot as plt
import random
import numpy as np


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list/np.sum(probability_list)


class Agent:
    def __init__(self, env, n=5, epsilon=0.4, initial_value=0.0, kappa=0.00001):
        self.env = env
        self.epsilon = epsilon
        self.value_state_action = collections.defaultdict(lambda: initial_value)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))
        self.model = collections.defaultdict(lambda: {})
        self.model_state_action_list = []
        self.n = n
        self.total_step_num = 0
        self.kappa = kappa
        self.initialize_q_and_model()

    def initialize_q_and_model(self):
        for state_iter in range(self.env.state_space.n):
            if state_iter in self.env.termination:
                continue
            for action_iter in range(self.env.action_space.n):
                new_state, reward, is_done, _ = self.env.step(action_iter, state=state_iter)
                self.model[state_iter][action_iter] = [reward, new_state, 0]
                self.model_state_action_list.append([state_iter, action_iter])

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def dyna_q_plus(self, number_of_episodes, alpha=0.1, gamma=1., epsilon=0.1):
        for epi_iter in range(number_of_episodes):
            state = self.env.reset()
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action, state=state)
                self.model[state][action] = [reward,new_state,self.total_step_num]
                # next state-action return
                # if not is_done:
                q_state_next = []
                for action_iter in range(self.env.action_space.n):
                    q_state_next.append(self.value_state_action[(new_state, action_iter)])
                q_state_next = max(q_state_next)
                q_state_current = self.value_state_action[(state, action)]
                self.value_state_action[(state, action)] = \
                    q_state_current + alpha * (reward + gamma * q_state_next - q_state_current)
                # update policy
                value_of_action_list = []
                possible_action_num = self.env.action_space.n
                for action_iter in range(possible_action_num):
                    value_of_action_list.append(self.value_state_action[(state, action_iter)])
                value_of_action_list = np.array(value_of_action_list)
                optimal_action = \
                    np.random.choice(np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                for action_iter in range(self.env.action_space.n):
                    if action_iter == optimal_action:
                        self.policies[state][action_iter] = 1 - epsilon + epsilon / possible_action_num
                    else:
                        self.policies[state][action_iter] = epsilon / possible_action_num
                # add into model
                # self.model[state][action] = [reward, new_state]
                # if [state, action] not in self.model_state_action_list:
                #     self.model_state_action_list.append([state, action])
                # planning
                if len(self.model_state_action_list) > 0:
                    for n_iter in range(self.n):
                        state_selected, action_selected = random.choice(self.model_state_action_list)
                        reward_in_model, new_state_in_model,last_update_time = \
                            self.model[state_selected][action_selected]
                        reward_in_model = reward_in_model + self.kappa * np.sqrt(self.total_step_num - last_update_time)
                        q_state_next_in_model = []
                        for action_iter in range(self.env.action_space.n):
                            q_state_next_in_model.append(self.value_state_action[(new_state_in_model, action_iter)])
                        q_state_next_in_model = max(q_state_next_in_model)
                        q_state_current_in_model = self.value_state_action[(state_selected, action_selected)]
                        self.value_state_action[(state_selected, action_selected)] = \
                            q_state_current_in_model + alpha * (reward_in_model + gamma * q_state_next_in_model -
                                                                q_state_current_in_model)
                        state = state_selected
                if is_done:
                    break
                self.total_step_num += 1
        return


if __name__ == '__main__':
    # env = GridWorld(6, [25,26,27,28,29], start_position=32, end_position_list=[5])
    # steps_matrix = np.zeros((3,50))
    # agent_n = [0,5,50]
    # repeat_n_times = 50
    # for j in range(repeat_n_times):
    #     for i in range(3):
    #         agent = Agent(env, n=agent_n[i])
    #         steps = agent.dyna_q(50, alpha=0.1, gamma=0.95,epsilon=.3)
    #         steps_matrix[i] += np.array(steps)
    # steps_matrix /= 50.
    # i = 0
    # for steps_array in steps_matrix:
    #     plt.plot(steps_array,label=agent_n[i])
    #     i += 1
    # plt.legend()
    # plt.show()
    env = GridWorld(6, [24, 25, 26, 27, 28], start_position=31, end_position_list=[5])
    agent = Agent(env, n=10)
    dq_step_rewards_list = []
    steps = 0
    for i in range(100):
        agent.dyna_q(1, alpha=0.1, gamma=0.95, epsilon=.3)
        dq_step_rewards_list.append([agent.total_step_num, i])
        if i == 50:
            agent.env = GridWorld(6, [25, 26, 27, 28, 29], start_position=31, end_position_list=[5])
    dq_step_rewards_list = np.array(dq_step_rewards_list)
    plt.plot(dq_step_rewards_list[:, 0], dq_step_rewards_list[:, 1], label='Dyna_Q')
    plt.legend()
    plt.show()
from Environment.gride_world import GridWorld
import numpy as np
import collections
import matplotlib.pyplot as plt
import random
import numpy as np
import TebularPlanningAndLearning.Dyna_Q as dq

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
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def dyna_q_p(self, number_of_episodes, alpha=0.1, gamma=1., epsilon=0.1):
        steps_used_in_episode = []
        for epi_iter in range(number_of_episodes):
            state = self.env.reset()
            step_nums = 0
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
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
                if [state, action] not in self.model_state_action_list:
                    self.model[state][action] = [reward, new_state]
                    self.model_state_action_list.append([state, action])
                elif self.model[state][action][1] != new_state:
                    self.model[state][action] = [reward, new_state]

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
                step_nums += 1
            steps_used_in_episode.append(step_nums)
        return steps_used_in_episode


if __name__ == '__main__':
    env = GridWorld(6, [24,25,26,27,28], start_position=31, end_position_list=[5])
    episode_numbers = 400
    dqp_step_rewards_list = []
    dqp_steps = 0
    dq_step_rewards_list = []
    dq_steps = 0
    agent_dqp = Agent(env, n=10)
    agent_dq = dq.Agent(env, n=10, initial_value=0.5)
    for i in range(episode_numbers):
        dqp_steps += agent_dqp.dyna_q_p(1, alpha=0.1, gamma=0.95, epsilon=.3)[0]
        dqp_step_rewards_list.append([dqp_steps, i])
        dq_steps += agent_dq.dyna_q(1, alpha=0.1, gamma=0.95, epsilon=.3)[0]
        dq_step_rewards_list.append([dq_steps, i])
        if i == episode_numbers/2:
            agent_dqp.env = GridWorld(6, [25, 26, 27, 28, 29], start_position=31, end_position_list=[5])
            agent_dq.env = GridWorld(6, [25, 26, 27, 28, 29], start_position=31, end_position_list=[5])
    dqp_step_rewards_list = np.array(dqp_step_rewards_list)
    plt.plot(dqp_step_rewards_list[:, 0], dqp_step_rewards_list[:, 1], label='Dyna_Q+')
    dq_step_rewards_list = np.array(dq_step_rewards_list)
    plt.plot(dq_step_rewards_list[:, 0], dq_step_rewards_list[:, 1], label='Dyna_Q')
    plt.legend()
    plt.show()

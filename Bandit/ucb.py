import numpy as np
import random
from environment.k_arm_bandit import KArmedBandit
import matplotlib.pyplot as plt
from rich.progress import track
from basic_moduls.ucb import UCB
from Bandit.epsilon_greedy import Agent as EG_Agent


class Agent():
    def __init__(self, env_, c_, initial_value_, step_size_):
        self._initial_value = initial_value_
        self._step_size = step_size_
        self._env = env_
        self._k = env_.action_space.n
        self._optimal_action = env_.optimal_action
        self._policy = UCB(c_)

    def select_action(self, action_value_, current_step_num_, action_selected_num_array_):
        action = self._policy(action_value_, self._k, current_step_num_, action_selected_num_array_)
        return action

    def run(self, total_step_num_, repeat_experiment_n_times_):
        average_reward = np.zeros(total_step_num_)
        optimal_action_percentage = np.zeros(total_step_num_)
        for _ in track(range(repeat_experiment_n_times_), description="Repeating Experiment..."):
            action_value_estimate = np.ones(self._k) * self._initial_value
            action_value_estimated_times = np.zeros(self._k)
            # environment reset. although it is useless here
            # keep it for a good habit
            state = self._env.reset()
            for step_i in range(total_step_num_):
                action = self.select_action(action_value_estimate, step_i+1, action_value_estimated_times)
                state, reward, is_done, _ = self._env.step(action)
                action_value_estimated_times[action] += 1
                # update
                if self._step_size == '1/n':
                    step_size = 1. / action_value_estimated_times[action]
                else:
                    step_size = self._step_size
                # pseudocode on page 32
                error_in_estimation = (reward - action_value_estimate[action])
                action_value_estimate[action] = action_value_estimate[action] + step_size * error_in_estimation
                average_reward[step_i] += reward
                if action in self._optimal_action():
                    optimal_action_percentage[step_i] += 1
        if repeat_experiment_n_times_ != 0:
            average_reward /= repeat_experiment_n_times_
            optimal_action_percentage /= repeat_experiment_n_times_
        return average_reward, optimal_action_percentage


# for figure 2.2
if __name__ == '__main__':
    env = KArmedBandit(10, np.random.normal(.0, 1.0, 10), np.ones(10))
    agent_ubc = Agent(env, 2, 0, '1/n')
    average_reward_ubc, optimal_action_percentage_0 = agent_ubc.run(1000, 2000)
    agent_epsilon_0_1 = EG_Agent(env, 0.1, 0, '1/n')
    average_reward_epsilon, optimal_action_percentage_epsilon = agent_epsilon_0_1.run(1000, 2000)
    plt.figure(1, figsize=(18, 10))
    plt.plot(average_reward_ubc, linewidth=1, alpha=0.7, c='g', label='UCB initial_value=0')
    plt.plot(average_reward_epsilon, linewidth=1, alpha=0.7, c='b', label='0.1-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()
    plt.savefig('./Figure/UCB_reward_F2.4.png')
    plt.figure(2, figsize=(18, 10))
    plt.plot(optimal_action_percentage_0, linewidth=1, alpha=0.7, c='g', label='UCB initial_value=0')
    plt.plot(optimal_action_percentage_epsilon, linewidth=1, alpha=0.7, c='b', label='0.1-greedy initial_value=0')

    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.savefig('./Figure/UCB_optimal_F2.4.png')
    plt.show()


import numpy as np
import random
from environment.k_arm_bandit import KArmedBandit
import matplotlib.pyplot as plt
from rich.progress import track
from basic_moduls.epsilon_greedy import EpsilonGreedy


class Agent:
    def __init__(self, env_, epsilon_, initial_value_, step_size_):
        self._initial_value = initial_value_
        self._step_size = step_size_
        self._env = env_
        self._k = env_.action_space.n
        self._optimal_action = env_.optimal_action
        self._policy = EpsilonGreedy(epsilon_)

    def select_action(self, action_value):
        prob_distribution = self._policy(action_value, self._k)
        action = np.random.choice(self._k, 1, p=prob_distribution)
        return action[0]

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
                action = self.select_action(action_value_estimate)
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
    agent_0 = Agent(env, 0, 0, '1/n')
    average_reward_0, optimal_action_percentage_0 = agent_0.run(1000, 2000)
    agent_0_1 = Agent(env, 0.1, 0, '1/n')
    average_reward_0_1, optimal_action_percentage_0_1 = agent_0_1.run(1000, 2000)
    agent_0_01 = Agent(env, 0.01, 0, '1/n')
    average_reward_0_01, optimal_action_percentage_0_01 = agent_0_01.run(1000, 2000)
    plt.figure(1)
    for data_i, c_i in zip([average_reward_0,
                            average_reward_0_1,
                            average_reward_0_01],
                           ['g', 'b', 'r']):
        plt.plot(data_i, linewidth=1, alpha=0.7, c=c_i, label='0-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()
    plt.savefig('./Figure/epsilon-greedy_F2.2.0.png')
    plt.figure(2)
    for data_i, c_i in zip([optimal_action_percentage_0,
                            optimal_action_percentage_0_1,
                            optimal_action_percentage_0_01],
                           ['g', 'b', 'r']):
        plt.plot(data_i, linewidth=1, alpha=0.7, c=c_i, label='0-greedy initial_value=0')

    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.savefig('./Figure/epsilon-greedy_F2.2.1.png')
    plt.show()


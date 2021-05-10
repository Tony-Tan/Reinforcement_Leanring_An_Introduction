# exercise 2.5 on page 33 of the 2nd edition Design and conduct an experiment to demonstrate the difficulties that
# sample-average methods have for non-stationary problems. Use a modified version of the 10-armed testbed in which
# all the q_{\star}(a) start out equal and then take independent random walks (say by adding a normally distributed
# increment with mean zero and standard deviation 0.01 to all the q_{\star}(a) on each step). Prepare plots like
# Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value
# method using a constant step-size parameter, \alpha= 0.1. Use \epsilon= 0.1 and longer runs, say of 10,000 steps.


import numpy as np
import random
from environment.k_arm_bandit import KArmedBanditRW
import matplotlib.pyplot as plt
from rich.progress import track
from basic_moduls.epsilon_greedy import EpsilonGreedy
from epsilon_greedy import Agent


if __name__ == '__main__':
    env = KArmedBanditRW(10, np.random.normal(.0, 1.0, 10), np.ones(10), 0, 0.01)
    agent_0 = Agent(env, 0, 0, 0.1)
    plt.figure(1)
    average_reward_0, optimal_action_percentage_0 = agent_0.run(10000, 100)
    plt.plot(average_reward_0, linewidth=1, alpha=0.7, c='b',
             label='0.1-greedy $\\alpha=0.1$ initial_value=0')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.figure(2)
    plt.plot(optimal_action_percentage_0, linewidth=1, alpha=0.7, c='b',
             label='0.1-greedy $\\alpha=0.1$ initial_value=0')
    plt.xlabel('Steps')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.show()



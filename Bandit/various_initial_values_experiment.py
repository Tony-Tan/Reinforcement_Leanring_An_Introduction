# Figure 2.3 shows the performance on the 10-armed bandit testbed of a greedy method using Q_1(a) = +5, for all a.

import numpy as np
from environment.k_arm_bandit import KArmedBandit
import matplotlib.pyplot as plt
from epsilon_greedy import Agent

if __name__ == '__main__':
    env = KArmedBandit(10, np.random.normal(.0, 1.0, 10), np.ones(10))
    agent_initial_q_0 = Agent(env, 0.1, 0, 0.1)
    _, optimal_action_percentage_initial_q_0 = agent_initial_q_0.run(1000, 2000)
    agent_initial_q_5 = Agent(env, 0, 5, 0.1)
    _, optimal_action_percentage_initial_q_5 = agent_initial_q_5.run(1000, 2000)
    plt.figure(1)
    plt.plot(optimal_action_percentage_initial_q_0, linewidth=1, alpha=0.7, c='g', label='0.1-greedy initial_value=0')
    plt.plot(optimal_action_percentage_initial_q_5, linewidth=1, alpha=0.7, c='b', label='greedy initial_value=5')
    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.savefig('./Figure/epsilon-greedy_initial_value_F2.3.png')
    plt.show()


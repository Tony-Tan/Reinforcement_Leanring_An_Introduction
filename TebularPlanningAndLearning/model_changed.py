import TebularPlanningAndLearning.Dyna_Q_plus as DQp
from Environment.gride_world import GridWorld
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = GridWorld(6, [24,25,26,27,28], start_position=31, end_position_list=[5])
    episode_numbers = 400
    dqp_step_rewards_list = []
    dqp_steps = 0
    dq_step_rewards_list = []
    dq_steps = 0
    agent_dqp = DQp.Agent(env, n=10, kappa=0.0001)
    agent_dq = DQp.Agent(env, n=10, kappa=0)
    dqp_reward = 0.0
    for i in range(episode_numbers):
        agent_dqp.dyna_q_plus(1, alpha=0.1, gamma=0.95, epsilon=.3)
        dqp_step_rewards_list.append([agent_dqp.total_step_num, i])
        agent_dq.dyna_q_plus(1, alpha=0.1, gamma=0.95, epsilon=.3)
        dq_step_rewards_list.append([agent_dq.total_step_num, i])
        if i == episode_numbers/4:
            agent_dqp.env = GridWorld(6, [25, 26, 27, 28], start_position=31, end_position_list=[5])
            agent_dq.env = GridWorld(6, [25, 26, 27, 28], start_position=31, end_position_list=[5])
    dqp_step_rewards_list = np.array(dqp_step_rewards_list)
    plt.plot(dqp_step_rewards_list[:, 0] - dqp_step_rewards_list[0][0], dqp_step_rewards_list[:, 1], label='Dyna_Q+')
    dq_step_rewards_list = np.array(dq_step_rewards_list)
    plt.plot(dq_step_rewards_list[:, 0] - dq_step_rewards_list[0][0], dq_step_rewards_list[:, 1], label='Dyna_Q')
    plt.legend()
    plt.show()
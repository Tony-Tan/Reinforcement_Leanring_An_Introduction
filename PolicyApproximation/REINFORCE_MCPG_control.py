from Environment.corridor_gridworld import ShortCorridor
import numpy as np
import matplotlib.pyplot as plt


class Linear_policy:
    """
    l(x,y) = 0*x + a*y + c
    """

    def __init__(self):
        self.weight = np.zeros(2)
        self.weight[1] = 1.

    def __call__(self, x, y):
        return self.weight.dot(np.array([x, y])) + 0.000001

    def derivative_ln(self, x, y):
        return np.array([0, y]) / self.__call__(x, y)


class Agent:
    def __init__(self, env):
        self.env = env
        self.policy = Linear_policy()

    def select_action(self, state):
        probability_distribution = []
        exp_sum = 0
        for action_iter in self.env.action_space:
            probability_distribution.append(self.policy(state, action_iter)*5.)
            exp_sum += np.exp(self.policy(state, action_iter)*5.)
        for i in range(len(probability_distribution)):
            probability_distribution[i] = np.exp(probability_distribution[i]) / exp_sum
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def play(self, number_of_episodes, alpha, gamma):
        reward_per_episode = []
        for eps_iter in range(number_of_episodes):
            reward_sum = 0
            episode = []
            state = self.env.reset()
            gamma_ = 1.
            total_g = 0
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
                episode.append([state, action, reward])
                total_g += gamma_ * reward
                gamma_ *= gamma
                reward_sum += reward
                if is_done:
                    break
                state = new_state
            reward_per_episode.append(reward_sum)
            gamma_t = 1.
            for epd_i in range(len(episode) - 1):
                state, action, r = episode[epd_i]
                g = total_g
                theta = self.policy.derivative_ln(state, action)
                self.policy.weight += alpha * gamma_t * g * theta
                gamma_t *= gamma
                total_g -= r
                total_g /= gamma
            # np.set_printoptions(precision=11)
            # print(eps_iter, self.policy.weight)
        return np.array(reward_per_episode)


if __name__ == '__main__':

    # for i in range(0, 1):

    episode_len = 1000
    repeat_time = 50
    steps = np.zeros(episode_len)
    for i in range(repeat_time):
        print('repeat time ' + str(i))
        env = ShortCorridor()
        agent = Agent(env)
        steps += agent.play(episode_len, 2e-7, 1)
    plt.plot(steps / repeat_time, alpha=0.7, label='$\\alpha=2e-7$')

    steps = np.zeros(episode_len)
    for i in range(repeat_time):
        print('repeat time ' + str(i))
        env = ShortCorridor()
        agent = Agent(env)
        steps += agent.play(episode_len, 2e-8, 1)
    plt.plot(steps / repeat_time, alpha=0.7, label='$\\alpha=2e-8$')

    steps = np.zeros(episode_len)
    for i in range(repeat_time):
        print('repeat time ' + str(i))
        env = ShortCorridor()
        agent = Agent(env)
        steps += agent.play(episode_len, 2e-9, 1)
    plt.plot(steps / repeat_time, alpha=0.7, label='$\\alpha=2e-9$')
    plt.legend()
    plt.show()

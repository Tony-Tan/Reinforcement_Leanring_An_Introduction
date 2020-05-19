import numpy as np
import matplotlib.pyplot as plt
import copy

def soft_max(x_0,x_array):
    denominator = 0
    for x_i in x_array:
        denominator += np.exp(x_i)

    return np.exp(x_0)/denominator


def random_select_action_by_preference(probability_of_action):
    length_of_preference = len(probability_of_action)
    integral_array = copy.deepcopy(probability_of_action)
    for i in range(1,length_of_preference):
        integral_array[i] += integral_array[i-1]
    random_value = np.random.randint(0,10000)/10000.
    for i in range(length_of_preference):
        if random_value > integral_array[i]:
            continue
        return i


class Bandit():
    def __init__(self, value_mean=0.0, value_var=1.0):
        """
        bandit, reward is produced by a normal distribution with mean and variance;
        :param value_mean: mean
        :param value_var: variance
        """
        self.value_mean = value_mean
        self.value_var = value_var

    def run(self):
        return np.random.normal(self.value_mean,self.value_var,1)[0]


class KArmedBandit():
    def __init__(self, bandit_generating_mean, bandit_generating_var, k):
        """
        k-armed bandit and its reward based on normal distribution with mean and variance as:
        :param bandit_generating_mean: the mean of the distribution of bandit
        :param bandit_generating_var: the variance of the distribution of bandit
        :param k: how many arms
        """
        self.k_armed_bandits = []
        self.greatest_value_bandit = -1
        self.k = k
        bandit_value_array = []
        for i in range(k):
            value_mean = np.random.normal(bandit_generating_mean, bandit_generating_var)
            bandit_value_array.append(value_mean)
            bandit = Bandit(value_mean, value_var=1.0)
            self.k_armed_bandits.append(bandit)
        self.greatest_value_bandit = np.argmax(np.array(bandit_value_array))

    def run(self):
        rewards = []
        for bandit in self.k_armed_bandits:
            rewards.append(bandit.run())
        return rewards

    def random_walk_update(self, random_walk_std_deviation):
        delta_mean = np.random.normal(0, random_walk_std_deviation, self.k)
        for i in range(len(self.k_armed_bandits)):
            self.k_armed_bandits[i].value_mean += delta_mean[i]
        bandit_value_array = []
        for i in range(len(self.k_armed_bandits)):
            bandit_value_array.append(self.k_armed_bandits[i].value_mean)
        self.greatest_value_bandit = np.argmax(np.array(bandit_value_array))


class KArmedBanditTestbed():
    def __init__(self, repeat_times, n_step,  bandit_mean, bandit_var, k, random_wark_std_deviation=0):
        """
        K-armed bandit testbed
        :param repeat_times:  how many time we will do the experiments to get an average
        :param n_step:  how many steps in each experiment
        :param bandit_mean: the mean of the distribution of each bandit
        :param bandit_var: the variance of the distribution of each bandit
        :param k: how many arms
        """
        self.n_step = n_step
        self.repeat_times = repeat_times
        self.bandit_mean = bandit_mean
        self.bandit_var = bandit_var
        self.k = k
        self.random_wark_std_deviation = random_wark_std_deviation

    def vare_greedy(self, varepsilon, initial_expectation, step_size_='1/n', greedy_method='None', c=2):
        """
        varepsilon-greedy method for different varepsilon
        when varepsilon  = 0 is the greedy method
        when greedy_method is setted to 'UCB' the UCB is employed
        :param varepsilon:  a array of varepsilong such as [0,0.1,0.01]
        :param initial_expectation: can ste all Q_1(a) to a constant
        :param greedy_method:  if it is set to 'UCB', employ the UCB method
        :param c: only useful when UCB method iw employed
        :return:
        """
        step_size = 0
        varepsilon_size = len(varepsilon)
        average_reward = np.zeros((varepsilon_size,self.n_step))
        percentage_of_optimal_action = np.zeros((varepsilon_size,self.n_step))

        for i in range(self.repeat_times):
            kab = KArmedBandit(self.bandit_mean, self.bandit_var, self.k)
            for vare_i in range(varepsilon_size):
                vare = varepsilon[vare_i]
                value_estimation = np.ones(self.k)*initial_expectation
                value_estimation_times = np.zeros(self.k)
                reward_per_step = np.zeros(self.n_step)
                optimal_action_or_not = np.zeros(self.n_step)
                for step_i in range(self.n_step):
                    # if this is a non-stationary task, the random walk standard deviation is setted
                    if self.random_wark_std_deviation != 0:
                        kab.random_walk_update(self.random_wark_std_deviation)
                    rewards = kab.run()
                    # exploration

                    if vare != 0 and np.random.randint(0, 1000) < 1000*vare:
                        action = np.random.randint(0, self.k)
                    else:
                        if greedy_method == 'UCB':
                            ucb_value_uncertainty = np.zeros(self.k)
                            for k_i in range(self.k):
                                if step_i == 0:
                                    break
                                if value_estimation_times[k_i] == 0:
                                    ucb_value_uncertainty[k_i] = 100000000
                                else:
                                    ucb_value_uncertainty[k_i] = c * np.sqrt(np.log(step_i)/value_estimation_times[k_i])
                            action = np.argmax(value_estimation + ucb_value_uncertainty)
                        else:
                            action = np.argmax(value_estimation)
                    # update
                    value_estimation_times[action] +=1

                    if step_size_ == '1/n':
                        step_size = 1/value_estimation_times[action]
                    else:
                        step_size = step_size_
                    error_in_estimation = (rewards[action]-value_estimation[action])
                    value_estimation[action] = value_estimation[action] + step_size*error_in_estimation

                    reward_per_step[step_i] = rewards[action]
                    if action == kab.greatest_value_bandit:
                        optimal_action_or_not[step_i] = 1

                average_reward[vare_i] += reward_per_step
                percentage_of_optimal_action[vare_i] += optimal_action_or_not
        average_reward /= self.repeat_times
        percentage_of_optimal_action /= self.repeat_times
        return average_reward, percentage_of_optimal_action

    def GBA(self, step_size, baseline_type='reward_mean'):

        average_reward = np.zeros(self.n_step)
        percentage_of_optimal_action = np.zeros(self.n_step)

        for i in range(self.repeat_times):
            preference_of_actions = np.zeros(self.k)
            pi_t_of_action = np.ones(self.k) / self.k
            kab = KArmedBandit(self.bandit_mean, self.bandit_var, self.k)
            r_mean = 0
            reward_per_step = np.zeros(self.n_step)
            optimal_action_or_not = np.zeros(self.n_step)
            for step_i in range(self.n_step):
                rewards = kab.run()
                action = random_select_action_by_preference(pi_t_of_action)
                r_t = rewards[action]
                r_mean = (r_t + step_i * r_mean)/(step_i+1)
                # Update preference
                if baseline_type == 'reward_mean':
                    delta_r = r_t - r_mean
                else:
                    delta_r = r_t - baseline_type
                preference_of_actions[action] += step_size*delta_r*(1 - pi_t_of_action[action])

                for action_i in range(self.k):
                    if action_i != action:
                        preference_of_actions[action_i] -= step_size * delta_r * pi_t_of_action[action_i]

                # Update probability of each action
                for action_i in range(self.k):
                    pi_t_of_action[action_i] = soft_max(preference_of_actions[action_i], preference_of_actions)

                reward_per_step[step_i] = rewards[action]
                if action == kab.greatest_value_bandit:
                    optimal_action_or_not[step_i] = 1

            average_reward += reward_per_step
            percentage_of_optimal_action += optimal_action_or_not
        average_reward /= self.repeat_times
        percentage_of_optimal_action /= self.repeat_times
        return average_reward, percentage_of_optimal_action


if __name__ == '__main__':
    """

    # optimal initial values
    kabt = KArmedBanditTestbed(2000, 200, 0, 1, 10)
    a_reward_greedy_i_5, p_optimal_action_greedy_i_5 = kabt.vare_greedy([0], 50, 0.1)
    a_reward_greedy_i_0, p_optimal_action_greedy_i_0 = kabt.vare_greedy([0.1], 0, 0.1)

    plt.figure(1)
    plt.plot(a_reward_greedy_i_0[0], linewidth=1, alpha =0.7,c='r', label='0.1-greedy initial_value=0')
    plt.plot(a_reward_greedy_i_5[0], linewidth=1, alpha =0.7,c='g', label='greedy initial_value=50')
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()
    plt.figure(2)
    plt.plot(p_optimal_action_greedy_i_0[0], linewidth=1, alpha=0.7, c='r', label='0.1-greedy initial_value=0')
    plt.plot(p_optimal_action_greedy_i_5[0], linewidth=1, alpha=0.7, c='g', label='greedy initial_value=50')
    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.show()
    
    # UCB
    kabt = KArmedBanditTestbed(2000, 200, 0, 1, 10)
    a_reward_greedy_UCB, p_optimal_action_greedy_UCB = kabt.vare_greedy([0], 0, greedy_method='UCB', c=2)
    a_reward_greedy, p_optimal_action_greedy = kabt.vare_greedy([0.1], 0)

    plt.figure(1)
    plt.plot(a_reward_greedy_UCB[0], linewidth=1, alpha=0.7, c='r', label='greedy UCB')
    plt.plot(a_reward_greedy[0], linewidth=1, alpha=0.7, c='g', label='0.1-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()
    plt.figure(2)
    plt.plot(p_optimal_action_greedy_UCB[0], linewidth=1, alpha=0.7, c='r', label='greedy UCB')
    plt.plot(p_optimal_action_greedy[0], linewidth=1, alpha=0.7, c='g', label='0.1-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.show()
    

    # 10-armed bandit
    kabt = KArmedBanditTestbed(200, 1000, 4, 1, 10)
    a_reward_greedy, p_optimal_action_greedy = kabt.vare_greedy([0,0.1,0.01], 0)

    plt.figure(1)
    plt.plot(a_reward_greedy[0], linewidth=1, alpha=0.7, c='r', label='greedy initial_value=0')
    plt.plot(a_reward_greedy[1], linewidth=1, alpha=0.7, c='g', label='0.1-greedy initial_value=0')
    plt.plot(a_reward_greedy[2], linewidth=1, alpha=0.7, c='b', label='0.01-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()
    plt.figure(2)
    plt.plot(p_optimal_action_greedy[0], linewidth=1, alpha=0.7, c='r', label='greedy initial_value=0')
    plt.plot(p_optimal_action_greedy[1], linewidth=1, alpha=0.7, c='g', label='0.1-greedy initial_value=0')
    plt.plot(p_optimal_action_greedy[2], linewidth=1, alpha=0.7, c='b', label='0.01-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.show()

 
    # non-stationary test with step-size=0.1 and distribution of reward with random walk of standard deviation
    # 0.01
    kabt = KArmedBanditTestbed(20, 10000, 0, 1, 10, 0.01)
    a_reward_greedy_1_n, p_optimal_action_greedy_1_n = kabt.vare_greedy([0.1], 0, '1/n')
    a_reward_greedy_c, p_optimal_action_greedy_c = kabt.vare_greedy([0.1], 0, 0.1)

    plt.figure(1)
    plt.plot(a_reward_greedy_1_n[0], linewidth=1, alpha=0.7, c='r',
             label='$\\varepsilon=0.1$ step_size=$\\frac{1}{n}$')
    plt.plot(a_reward_greedy_c[0], linewidth=1, alpha=0.7, c='g',
             label='$\\varepsilon=0.1$ step_size=$0.1$')
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()
    plt.figure(2)
    plt.plot(p_optimal_action_greedy_1_n[0], linewidth=1, alpha=0.7, c='r',
             label='$\\varepsilon=0.1$ step_size=$\\frac{1}{n}$')
    plt.plot(p_optimal_action_greedy_c[0], linewidth=1, alpha=0.7, c='g',
             label='$\\varepsilon=0.1$ step_size=$0.1$')
    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.show()
    """
    # Gradient bandit algorithm
    step_num = 1000
    kabt = KArmedBanditTestbed(1000, step_num, 4, 1, 10)
    a_reward_greedy_gba, p_optimal_action_greedy_gba = kabt.GBA(0.1,-4)
    a_reward_greedy_gba_04, p_optimal_action_greedy_gba_04 = kabt.GBA(0.4,-4)

    a_reward_greedy_gba_base_4, p_optimal_action_greedy_gba_base_4 = kabt.GBA(0.1)
    a_reward_greedy_gba_base_4_04, p_optimal_action_greedy_gba_base_4_04 = kabt.GBA(0.4)
    # plt.figure(1)
    # plt.plot(a_reward_greedy_gba, linewidth=1, alpha=0.3, c='g',
    #          label='GBA step_size= 0.1')
    # plt.plot(a_reward_greedy_gba_04, linewidth=1, alpha=0.3, c='b',
    #          label='GBA step_size= 0.4')
    # plt.plot(a_reward_greedy_gba_base_4, linewidth=1, alpha=0.7, c='g',
    #          label='GBA step_size= 0.1')
    # plt.plot(a_reward_greedy_gba_base_4_04, linewidth=1, alpha=0.7, c='b',
    #          label='GBA step_size= 0.4')
    # plt.plot(a_reward_greedy_c[0], linewidth=1, alpha=0.3, c='gray',
    #          label='$\\varepsilon=0.1$ step_size=0.1')
    # plt.xlabel('time')
    # plt.ylabel('reward')
    # plt.legend()
    plt.figure(2)
    plt.plot(p_optimal_action_greedy_gba, linewidth=1, alpha=0.3, c='g')
    plt.text(step_num / 2, p_optimal_action_greedy_gba[int(step_num/2)], '$\\alpha$= 0.1')

    plt.plot(p_optimal_action_greedy_gba_04, linewidth=1, alpha=0.3, c='b')
    plt.text(step_num / 2, p_optimal_action_greedy_gba_04[int(step_num / 2)], '$\\alpha$= 0.4')
    plt.text(step_num / 2, 0.5 * (p_optimal_action_greedy_gba[int(step_num / 2)] +
                                  p_optimal_action_greedy_gba_04[int(step_num / 2)]), 'without baseline')

    plt.plot(p_optimal_action_greedy_gba_base_4, linewidth=1, alpha=0.7, c='g')
    plt.text(step_num / 2, p_optimal_action_greedy_gba_base_4[int(step_num / 2)], '$\\alpha$= 0.1 ')

    plt.plot(p_optimal_action_greedy_gba_base_4_04, linewidth=1, alpha=0.7, c='b')
    plt.text(step_num / 2, p_optimal_action_greedy_gba_base_4_04[int(step_num / 2)], '$\\alpha$= 0.4')
    plt.text(step_num / 2, 0.5*(p_optimal_action_greedy_gba_base_4_04[int(step_num / 2)]+
                                p_optimal_action_greedy_gba_base_4[int(step_num / 2)]), 'with baseline')

    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    #plt.legend()
    plt.show()

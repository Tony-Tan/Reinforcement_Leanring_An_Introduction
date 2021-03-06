import collections
from Environment.mountain_car import MountainCar
import numpy as np
import matplotlib.pyplot as plt
import random
import copy


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class LinearFunction:
    def __init__(self, n):
        self.n = n
        self.weight = np.zeros(n)

    def __call__(self, x):
        sum = 0
        for i in x:
            sum += self.weight[i]
        return sum


class Agent:
    def __init__(self, environment_):
        self.env = environment_
        self.tiling_block_num = 8
        self.tiling_num = 8
        self.size_of_weights = self.tiling_num * self.tiling_block_num * self.tiling_block_num
        self.value_of_state_action = LinearFunction(self.size_of_weights * 4)
        # parameters for feature extraction
        width = self.env.position_bound[1] - self.env.position_bound[0]
        height = self.env.velocity_bound[1] - self.env.velocity_bound[0]
        self.block_width = width / (self.tiling_block_num - 1)
        self.block_height = height / (self.tiling_block_num - 1)
        self.width_step = self.block_width / self.tiling_num
        self.height_step = self.block_height / self.tiling_num
        self.tiling_dict = {}
        # self.block_start_position = []
        # for i in range(self.tiling_num):
        #     x = self.width_step * random.uniform(0, 1)
        #     y = self.height_step * random.uniform(0, 1)
        #     self.block_start_position.append((x, y))

    def state_feature_extract(self, state, action):
        position, velocity = state
        feature = []
        x = position - self.env.position_bound[0]
        y = velocity - self.env.velocity_bound[0]
        for i in range(self.tiling_num):
            # x_ = x - self.block_start_position[i][0]
            # y_ = y - self.block_start_position[i][1]
            x_ = x - i * self.width_step
            y_ = y - i * self.height_step
            x_position = int(x_ / self.block_width + 1)
            y_position = int(y_ / self.block_height + 1)
            org_feature = (i * self.tiling_block_num * self.tiling_block_num +
                           y_position * self.tiling_block_num + x_position) * 4 + action + 1

            if org_feature in self.tiling_dict.keys():
                feature.append(self.tiling_dict[org_feature])
            else:
                self.tiling_dict[org_feature] = len(self.tiling_dict.keys())
                feature.append(self.tiling_dict[org_feature])
        return feature

    # def active_features(self, state_feature, action):
    #     active_features_index = copy.deepcopy(state_feature)
    #     active_features_index.append(self.size_of_weights + action)
    #     return active_features_index

    def select_action(self, state, epsilon=0.1):
        value_of_action_list = []
        policies = np.zeros(self.env.action_space.n)
        for action_iter in range(self.env.action_space.n):
            state_action_feature = self.state_feature_extract(state, action_iter)
            value_of_action_list.append(self.value_of_state_action(state_action_feature))
        value_of_action_list = np.array(value_of_action_list)
        optimal_action = np.random.choice(
            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        for action_iter in range(self.env.action_space.n):
            if action_iter == optimal_action:
                policies[action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
            else:
                policies[action_iter] = epsilon / self.env.action_space.n
        probability_distribution = policies
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def running(self, iteration_times, alpha=0.01, gamma=0.9, lambda_coe=0.9):
        total_step = []
        for iteration_time in range(iteration_times):
            step_num = 0
            eligibility_trace = np.zeros(self.size_of_weights * 4)
            state = self.env.reset()
            action = self.select_action(state)
            # get reward R and next state S'
            while True:
                next_state, reward, is_done, _ = self.env.step(action)

                step_num += 1
                delta = reward
                state_action_feature = self.state_feature_extract(state, action)
                for idx_of_feature in state_action_feature:
                    delta -= self.value_of_state_action.weight[idx_of_feature]
                    # eligibility_trace[idx_of_feature] += 1
                    eligibility_trace[idx_of_feature] = 1

                if is_done:
                    self.value_of_state_action.weight += alpha * delta * eligibility_trace
                    break

                next_action = self.select_action(next_state)
                next_active_features = self.state_feature_extract(next_state, next_action)
                for idx_of_feature in next_active_features:
                    delta += gamma * self.value_of_state_action.weight[idx_of_feature]

                self.value_of_state_action.weight += alpha * delta * eligibility_trace
                eligibility_trace = gamma * lambda_coe * eligibility_trace

                state = next_state
                action = self.select_action(state)

            total_step.append(step_num)
            print(iteration_time, step_num)
        return np.array(total_step)


if __name__ == '__main__':

    repeat_times = 10
    step_num_list = np.zeros(50)
    for _ in range(repeat_times):
        env = MountainCar()
        print('1 round ' + str(_))
        agent = Agent(env)
        step_num_list += agent.running(50, alpha=0.01, lambda_coe=0.68)
    plt.plot(step_num_list / float(repeat_times), c='g', alpha=0.7, label='$\\alpha$=0.1/8')

    # step_num_list = np.zeros(100)
    # for _ in range(repeat_times):
    #     print('2 round ' + str(_))
    #     agent = Agent(env)
    #     step_num_list += agent.running(100, alpha=0.2 / 8.)
    # plt.plot(step_num_list / float(repeat_times), c='b', alpha=0.7, label='$\\alpha$=0.2/8')
    #
    # step_num_list = np.zeros(100)
    # for _ in range(repeat_times):
    #     print('3 round ' + str(_))
    #     agent = Agent(env)
    #     step_num_list += agent.running(100, alpha=0.5 / 8.)
    # plt.plot(step_num_list / float(repeat_times), c='r', alpha=0.7, label='$\\alpha$=0.5/8')

    plt.legend()
    plt.show()

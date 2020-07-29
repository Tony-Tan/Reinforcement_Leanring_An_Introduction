import collections
from Environment.mountain_car import MountainCar
import numpy as np
import matplotlib.pyplot as plt


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class LinearFunction:
    def __init__(self, n):
        self.weight = np.zeros(n+1)

    def __call__(self, x_1, x_2):
        x = [i for i in x_1]
        x.append(x_2)
        x = np.array(x)
        return self.weight.transpose().dot(x)

    def derivative(self,  x_1, x_2):
        x = [i for i in x_1]
        x.append(x_2)
        x = np.array(x)
        return x


class Agent:
    def __init__(self, environment_):
        self.env = environment_
        self.tiling_block_num = 8
        self.tiling_num = 8
        self.value_of_state_action = LinearFunction(self.tiling_num)
        # parameters for feature extraction
        width = self.env.position_bound[1] - self.env.position_bound[0]
        height = self.env.velocity_bound[1] - self.env.velocity_bound[0]
        self.block_width = width / (self.tiling_block_num - 1)
        self.block_height = height / (self.tiling_block_num - 1)
        self.width_step = self.block_width / self.tiling_num
        self.height_step = self.block_height / self.tiling_num

    def state_feature_extract(self, state):
        position, velocity = state
        feature = np.zeros(self.tiling_num)
        x = position - self.env.position_bound[0]
        y = velocity - self.env.velocity_bound[0]
        for i in range(self.tiling_num):
            x_ = x - i * self.width_step
            y_ = y - i * self.height_step
            x_position = int(x_ / self.block_width) + 1
            y_position = int(y_ / self.block_height) + 1
            feature[i] = y_position * self.tiling_block_num + x_position
        return feature

    def select_action(self, state_feature, epsilon=0.1):
        value_of_action_list = []
        policies = np.zeros(self.env.action_space.n)
        for action_iter in range(self.env.action_space.n):
            value_of_action_list.append(self.value_of_state_action(state_feature, action_iter))
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

    def running(self, iteration_times, alpha=0.00001, gamma=0.9):
        for iteration_time in range(iteration_times):
            state = self.env.reset()
            state_feature = self.state_feature_extract(state)
            action = self.select_action(state_feature)
            # get reward R and next state S'
            while True:
                next_state, reward, is_done, _ = self.env.step(action)
                next_state_feature = self.state_feature_extract(next_state)
                if is_done:
                    self.value_of_state_action.weight += \
                        alpha * (reward - self.value_of_state_action(state_feature, action)) * \
                        self.value_of_state_action.derivative(state_feature, action)
                    break
                next_action = self.select_action(next_state_feature)
                self.value_of_state_action.weight += \
                    alpha * (reward + gamma * self.value_of_state_action(next_state_feature, next_action)
                             - self.value_of_state_action(state_feature, action)) * \
                    self.value_of_state_action.derivative(state_feature, action)
                action = next_action
                state_feature = next_state_feature


if __name__ == '__main__':
    env = MountainCar()
    agent = Agent(env)
    agent.running(1)


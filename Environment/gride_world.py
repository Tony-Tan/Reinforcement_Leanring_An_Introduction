import collections
import matplotlib.pyplot as plt


class Space:
    def __init__(self, n_list):
        self.n_list = n_list
        self.n = len(n_list)

    def __getitem__(self, item):
        return self.n_list[item]


class GridWorld:
    def __init__(self, n=4):
        self.grid_size = n
        self.state_space = Space([i for i in range(n*n)])
        self.action_space = Space([0, 1, 2, 3])
        self.current_state = -1
        self.termination = [n*n - 1]

    def reset(self):
        # self.current_state = self.state_space[int(self.grid_size * self.grid_size / 2)]
        self.current_state = self.state_space[0]
        return self.current_state


    def step(self, action):
        height = width = self.grid_size
        # punish_value = -10
        punish_value = -10
        if action == 0:
            # go left
            new_position = self.current_state % width - 1
            if new_position < 0:
                return None, punish_value, True, {}
            else:
                self.current_state -= 1
                if self.current_state in self.termination:
                    return self.current_state, 1, True, {}
                else:
                    return self.current_state, 0, False, {}

        if action == 1:
            # go upper
            new_position = int(self.current_state / width) - 1
            if new_position < 0:
                return None, punish_value, True, {}
            else:
                self.current_state -= width
                if self.current_state in self.termination:
                    return self.current_state, 1, True, {}
                else:
                    return self.current_state, 0, False, {}

        if action == 2:
            # go right
            new_position = self.current_state % width + 1
            if new_position >= width:
                return None, punish_value, True, {}
            else:
                self.current_state += 1
                if self.current_state in self.termination:
                    return self.current_state, 1, True, {}
                else:
                    return self.current_state, 0, False, {}

        if action == 3:
            # go down
            new_position = int(self.current_state / width) + 1
            if new_position >= height:
                return None, punish_value, True, {}
            else:
                self.current_state += width
                if self.current_state in self.termination:
                    return self.current_state, 1, True, {}
                else:
                    return self.current_state, 0, False, {}

    def plot_grid_world(self, policy):
        n = self.grid_size
        size_of_action_space = self.action_space.n
        plt.figure(figsize=(8, 8))
        for i in range(n):
            for j in range(n):
                if i*n+j in self.termination:
                    plt.text((j + 1.05) / (n + 1), 1 - (i + 0.95) / (n + 1), 'End',
                             size='medium')
                    continue
                for action_iter in range(size_of_action_space):
                    if policy[i*n+j][action_iter] != 0:
                        length = policy[i*n+j][action_iter]
                        if action_iter == 0:
                            plt.arrow((j + 1) / (n + 1), 1 - (i + 1) / (n + 1),  -1*length/ (4 * (n + 1)), 0,
                                      head_width=0.01, head_length=0.01, fc='k', ec='k')
                        if action_iter == 1:
                            plt.arrow((j + 1) / (n + 1), 1 - (i + 1) / (n + 1),  0, 1*length / (4 * (n + 1)),
                                      head_width=0.01, head_length=0.01, fc='k', ec='k')
                        if action_iter == 2:
                            plt.arrow((j + 1) / (n + 1), 1 - (i + 1) / (n + 1),  +1*length/ (4 * (n + 1)), 0,
                                      head_width=0.01, head_length=0.01, fc='k', ec='k')
                        if action_iter == 3:
                            plt.arrow((j + 1) / (n + 1), 1 - (i + 1) / (n + 1),  0, -1*length / (4 * (n + 1)),
                                      head_width=0.01, head_length=0.01, fc='k', ec='k')
        plt.show()


if __name__ == '__main__':
    policy_test = collections.defaultdict(lambda: [.3,.2,.05,.45])
    env = GridWorld(10)
    env.plot_grid_world(policy_test)
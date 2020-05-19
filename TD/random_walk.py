import numpy


class Space:
    def __init__(self, initial_list):
        self.list = initial_list
        self.n = len(initial_list)

    def __getitem__(self, index):
        return self.list[index]


class RandomWalk:
    def __init__(self):
        self.state_space = Space([-3, -2, -1, 0, 1, 2, 3])
        self.action_space = Space([-1, +1])
        self.current_state = 3

    def reset(self):
        self.current_state = 3
        return 3

    def step(self, action):
        state = self.current_state + self.action_space[action]
        if self.state_space[state] == -3:
            return None, 0, True, {}
        elif self.state_space[state] == 3:
            return None, 1, True, {}
        else:
            self.current_state = state
            return state, 0, False, {}


if __name__ == '__main__':
    env = RandomWalk()
    print(env.state_space[0])
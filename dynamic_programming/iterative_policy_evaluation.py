# Example 4.1 4x4 gridworld shown below
# the methods we using here is Dynamic Programming on page 75
# not in-place methods

import copy
import numpy as np
from environment.grid_world4x4 import GridWorld

class Agent:
    def __init__(self, env_, gamma_=1):
        self.__env = env_
        self.__gamma = gamma_
        self.state_values_func = np.zeros(env_.state_space.n)
        self.state_values_func_updated = np.zeros(env_.state_space.n)
        self.__policy = np.ones(env_.action_space.n) / env_.action_space.n

    def predict(self):
        repeat_i = 0
        while True:
            self.state_values_func = copy.deepcopy(self.state_values_func_updated)
            for state_i in self.__env.state_space:
                sum_ = 0.0
                for action_i in self.__env.action_space:
                    self.__env.set_current_state(state_i)
                    next_state, reward, _, _ = self.__env.step(action_i)
                    sum_ += self.__policy[action_i] * (reward + self.__gamma * self.state_values_func[next_state])
                self.state_values_func_updated[state_i] = sum_

            # condition to stop the iteration
            # delta is small enough
            if np.min(np.abs(self.state_values_func_updated - self.state_values_func)) < 0.000001 and repeat_i > 1000:
                print('Prediction has finished! in %d loops'%repeat_i)
                return True
            repeat_i += 1


if __name__ == '__main__':
    env = GridWorld()
    agent = Agent(env)
    agent.predict()
    print(agent.state_values_func.reshape([4, 4]).round(1))

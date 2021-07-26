# the environment on page 84

import environment.environment_template as et
from basic_classes import Space
import random
import numpy as np


class GamblerProblem(et.ENV):
    def __init__(self, p_, initial_capital_):
        super(GamblerProblem).__init__()
        # the probability to come up head
        self.p = p_
        self.initial_capital = initial_capital_
        self.action_space = Space(range(0, initial_capital_))
        self.state_space = 100

    def reset(self):
        self.action_space = Space(range(0, self.initial_capital))

    def step(self, action_):
        if random.random() < self.p:
            self.state_space += action_
            if self.state_space >= 100:
                return self.state_space, 1, True, {}
            else:
                return self.state_space, 0, False, {}
        else:
            self.state_space -= action_
            return self.state_space, 0, False, {}


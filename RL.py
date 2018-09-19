# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

class Q_lam():
    def __init__(self, states, actions, epsilon=0.9, gamma=0.9, lam=0.5):
        self.states = states
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam

        columns = pd.Series(actions)
        index = pd.Series(states)

        self.q_table = pd.DataFrame(0, columns=columns, index=index)
        self.e_table = pd.DataFrame(0, columns=columns, index=index)

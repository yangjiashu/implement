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
        print(self.q_table)
        print(self.e_table)
    def e_greedy(self, s):
        # s[1]表示剩余座位，s[2]表示剩余售票时间
        # exploitation
        if np.random.uniform() < self.epsilon:
            Q_values = self.q_table.loc[s, :]
            Q_values = Q_values.reindex(np.random.permutation(Q_values.index))
            action = Q_values.argmax()
        # exploration
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, done, s, a, r, s_, alpha):
        if done:
            q_target = r
        else:
            Q_values = self.q_table.loc[s_, :]
            Q_values = Q_values.reindex(np.random.permutation(Q_values.index))
            a_best = Q_values.argmax()

            Q_target = r + self.gamma * self.q_table.loc[s_, a_best]
            TD_error = Q_target - self.q_table.loc[s,a]

            self.e_table.loc[s,a] = 1

            self.q_table = self.q_table + alpha * self.e_table * TD_error

            a_ = self.e_greedy(s_)
            if a_ == a_best:
                self.e_table = self.lam * self.e_table
            else:
                self.e_table = 0

        return s_, a_
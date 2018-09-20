# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

class Q_lam():
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

        columns = pd.Series(actions)
        index = pd.Series(states)

        self.q_table = pd.DataFrame(0, columns=columns, index=index)
        self.e_table = pd.DataFrame(0, columns=columns, index=index)

    def choose_action(self, state, epsilon):

        # 提取出index=state的一行getrow，并找出该行最大Q值的action，即最大值对应的下标
        getrow = self.q_table.loc[[state], :].ix[0]
        getrow = getrow.reindex(np.random.permutation(getrow.index))
        action = getrow.idxmax()

        # if geedy
        if np.random.uniform() > epsilon:
            greedy = True
            return action, greedy

        # if not greedy
        else:
            newrow = getrow.drop(action)
            action = np.random.choice(newrow)
            greedy = False
            return action, greedy

    def learn(self, s, a, r, s_, a_, done, alpha=0.5, gamma=0.9, lam=0.8):
        # step-1 选择状态s_的最优策略
        a_star, greedy = self.choose_action(s_, 0)

        # step-2 count Q-target
        # step-2.1
        if not done:
            target = r + gamma * self.q_table.loc[[s_], a_star].iloc[0]

        # step-2.2
        else:
            target = r

        # step-3 count TD-error
        error = target - self.q_table.loc[[s], a].iloc[0]

        # step-4 update the tables
        self.e_table.loc[[s], a] = 1
        self.q_table = self.q_table + alpha * error * self.e_table

        # step-5 update e_table
        if a_star == a_:
            self.e_table = self.e_table * lam
        else:
            self.e_table[:] = 0


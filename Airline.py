# coding: UTF-8
import pandas as pd
import numpy as np
import gym

class Airline(gym.Env):
    def __init__(self, capacity, times):
        '''
        :param capacity: An integer, representing the total capacity of the airplane
        :param time: An ndarray, a set of time horizon
        '''
        self.capacity = capacity
        self.times = times

        self.left_c = capacity
        self.i = 0
        self.left_t = self.times[self.i]

    def reset(self):
        self.left_c = self.capacity
        self.i = 0
        self.left_t = self.times[self.i]

        return tuple([self.left_c, self.left_t])

    def step(self, action):
        # Firstly, change the environment to next state.
        # step-1
        order = int(round(self.demand(action, self.left_t)))

        # terminal
        # 如果时间处在最后一个定价时间点或者需求大于库存，则下一个状态为terminal
        if self.i == len(self.times) - 1 or order - self.left_c >= 0:
            # step-2.21
            s_ = tuple([0, 0])
            r = action * min(self.left_c, order)
            done = True
            info = {}

        # not terminal
        else:
            # step-2.11 更行environment的状态
            self.left_c = self.left_c - order
            self.i += 1
            self.left_t = self.times[self.i]

            # step-2.12 获取观测值
            s_ = tuple([self.left_c,self.left_t])
            r = order * action
            done = False
            info = {}

        # Then return the observations
        return s_, r, done, info

    def demand(self, a, t):
        return 2.5 - 0.5 * t * np.exp(-1.5 / (t * a))
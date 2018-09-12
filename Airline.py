# coding: UTF-8
import pandas as pd
import numpy as np
import gym

class Airline(gym.Env):
    def __init__(self, capacity, time):
        '''
        :param capacity: An integer, representing the total capacity of the airplane
        :param time: An ndarray, a set of time horizon
        '''
        self.capacity = capacity
        self.time = time

        self.left_c = capacity #现在的剩余座位数是总容量
        self.left_t = time[len(time)-1] #现在的剩余时间是销售时间
        self.i = 0 #time数组的下标
    def reset(self):
        self.left_c = self.capacity
        self.i = 0
        self.left_t = self.time[self.i]

        return (self.left_c, self.left_t)

    def step(self, action):
        order = self.demmand_func(self.left_t, action)
        #terminal
        if (self.left_c - order) <= 0 or self.i == len(self.time)-1:
            left_c_ = max(0, self.left_c-order)
            #return state, reward, done, info
            return (self.left_c, self.left_t), min(order, self.left_c) * action, \
                    1, {}

        #not terminal
        else:
            left_c_ = self.left_c - order #求出s_
            self.left_c = left_c_ # 令s = s_
            self.i += 1
            self.left_t = self.time[self.i]
            return (self.left_c, self.left_t), order * action, \
                    0, {}

    def demmand_func(self, t, a):
        return 75-5*t*np.exp(-2/t * a/100)
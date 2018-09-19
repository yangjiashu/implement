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


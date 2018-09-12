# -*- coding: UTF-8 -*-
from Airline import Airline
from RL import Q_lam
import numpy as np

def update(env, RL, iter_times):
    for i in range(iter_times):
        RL.e_table *= 0
        s = env.reset()
        a = RL.e_greedy(s)

        while True:
            alpha = 1 / (i + 2)
            s_, r, done, info = env.step(a)
            s, a = RL.learn(done, s, a, r, s_, alpha)

            if done:
                break

        if i % 50 == 0:
            print(RL.q_table)


if __name__ == '__main__':
    capacity = 5
    time = [10, 7, 4, 1]

    states = []
    for i in range(capacity):
        for t in time:
            states.append(tuple([i+1,t]))
    print(states)
    prices = np.linspace(70, 120, 3)
    iter_times = 1000

    env = Airline(capacity, time)
    RL = Q_lam(states,prices)

    update(env, RL, iter_times)

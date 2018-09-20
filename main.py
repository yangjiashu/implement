# -*- coding: UTF-8 -*-
import numpy as np
from Airline import *
from Q_lam import *

if __name__ == '__main__':
    # block-1 initial
    capacity = 10
    times = np.array([5, 4, 3, 2, 1])
    states = []
    for c in range(capacity):
        for t in times:
            states.append(tuple([c+1, t]))

    actions = [0.2, 0.4, 0.6, 0.8, 1.0]

    env = Airline(capacity, times)
    RL = Q_lam(states, actions)
    iter_times = 1000

    # block-2 iteration
    for i in range(iter_times + 1):
        s = env.reset()
        a = np.random.choice(actions)

        while True:
            s_, r, done, info = env.step(a)

            # not terminal
            if not done:
                a_, greedy = RL.choose_action(s_, 0.1)
                RL.learn(s, a, r, s_, a_, done)
                s = s_
                a = a_
            # terminal
            else:
                a_ = a
                RL.learn(s, a, r, s_, a_, done)
                break
        if i % 50 == 0:
            print(RL.q_table)
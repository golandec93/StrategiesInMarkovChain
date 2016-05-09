# the process of consecutive decisions
import numpy as np
from src.utils import *


number_of_states = 4
states = generate_states(number_of_states)
number_of_iterations = 10

print_states(states)


behavior = []
v_old = np.empty(number_of_states)
for step in np.arange(0, number_of_iterations):
    d = np.empty(number_of_states)
    v_new = np.empty(number_of_states)
    for i in np.arange(0, number_of_states):
        strategies, costs = states[i]
        number_of_strategies = len(strategies)
        v_i = np.empty(number_of_strategies)

        for k in np.arange(0, number_of_strategies):
            if step == 0:
                v_i[k] = np.sum(strategies[k]*costs[k])
            else:
                v_i[k] = np.sum(strategies[k]*costs[k]) + sum(strategies[k]*v_old)

        v_new[i] = np.max(v_i)
        d[i] = np.argmax(v_i) + 1

    behavior.append(d)
    v_old = v_new

print('\n')
for d in behavior:
    print(d)
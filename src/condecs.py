# the process of consecutive decisions
import numpy as np
from src.utils import states_to_string


def eval_behavior(states, number_of_iterations=10, debug_enabled=False):
    def debug(obj):
        if debug_enabled:
            print('debug: ' + str(obj))

    debug(states_to_string(states))
    number_of_states = len(states)
    behavior = []
    income = []
    v_old = np.zeros(number_of_states)

    '''# evaluating for 1 step of algorithm
    d = np.empty(number_of_states)
    v_new = np.empty(number_of_states)
    for i in np.arange(0, number_of_states):
        strategies, costs = states[i]
        number_of_strategies = len(strategies)
        v_i = np.empty(number_of_strategies)
        for k in np.arange(0, number_of_strategies):
            v_i[k] = np.sum(strategies[k] * costs[k])
        v_new[i] = np.max(v_i)
        d[i] = np.argmax(v_i) + 1
    income.append(v_new)
    behavior.append(d)'''

    for step in np.arange(0, number_of_iterations):
        d = np.empty(number_of_states)
        v_new = np.empty(number_of_states)
        for i in np.arange(0, number_of_states):
            strategies, costs = states[i]
            number_of_strategies = len(strategies)
            v_i = np.empty(number_of_strategies)

            for k in np.arange(0, number_of_strategies):
                if step == 0:
                    v_i[k] = np.sum(strategies[k] * costs[k])
                else:
                    v_i[k] = np.sum(strategies[k] * costs[k]) + sum(strategies[k] * v_old)

            v_new[i] = np.max(v_i)
            d[i] = np.argmax(v_i) + 1

        behavior.append(d)
        income.append(v_new)
        v_old = v_new

    debug("behavior")
    for d in behavior:
        debug(d)
    debug("income")
    for inc in income:
        debug(inc)

    return behavior, income

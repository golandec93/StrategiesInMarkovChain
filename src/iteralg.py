import numpy as np
from src.utils import get_q


def eval_behavior(states, debug_enabled=False):
    def debug(*objects):
        if debug_enabled is True:
            res = ''
            for obj in objects:
                res += str(obj) + " "
            print("[DEBUG] " + res)

    # init data
    q = get_q(states)
    number_of_states = len(states)

    # initial step
    x = np.ndarray(number_of_states, int)
    current_income = np.ndarray(number_of_states)
    for i in np.arange(0, number_of_states):
        x[i] = (np.argmax(q[i]))
        current_income[i] = max(q[i])

    stop_flag = 0
    iterations = 0
    behavior = [x]
    income = [current_income]
    while stop_flag != 1 and iterations < 100:
        # step 1 parameters: states, x
        debug("iteration: ", iterations)
        p_matrix = np.ndarray((number_of_states, number_of_states))
        q_cur = np.zeros(number_of_states)
        for i in np.arange(0, number_of_states):
            q_cur[i] = -q[i][x[i]]
            probs, costs = states[i]
            p_matrix[i] = probs[x[i]]
            p_matrix[i][number_of_states - 1] = -1
            p_matrix[i][i] -= 1
        debug("p_matrix:\n", p_matrix)
        v = np.linalg.solve(p_matrix, q_cur)
        g = v[number_of_states - 1]
        v[number_of_states - 1] = 0
        debug("v:\n", v)
        debug('g:', g)

        # step 2  q_i_k + sum(p_i_j*v_j)
        new_x = np.ndarray(number_of_states, int)
        for i in range(0, number_of_states):
            debug("state: ", i)
            p, c = states[i]
            number_of_strategies = len(p)
            res = np.ndarray(number_of_strategies)
            for k in range(0, number_of_strategies):
                res[k] = q[i][k] + np.sum(v * p[k])
            debug("income for each strategy:\n", res)
            current_income[i] = max(res)
            new_x[i] = res.argmax()

        if np.array_equal(x, new_x):
            stop_flag += 1
        iterations += 1
        behavior.append(new_x + 1)
        income.append(current_income)
        x = new_x

    debug("behavior", behavior)
    debug("income", income)
    return behavior, income

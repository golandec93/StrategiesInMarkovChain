import numpy as np
from src.utils import generate_states
from src.utils import print_states
from src.utils import get_sample


def get_q(states):
    q = {}
    number_of_states = len(states)
    for i in range(0, number_of_states):
        probs, costs = states[i]
        number_of_stratagies = len(probs)
        qq = np.ndarray(number_of_stratagies)
        for j in np.arange(0, number_of_stratagies):
            qq[j] = (np.sum(probs[j] * costs[j]))
        q[i] = qq
    return q


# init data
# number_of_states = 4
# states = generate_states(number_of_states)
states = get_sample()
q = get_q(states)
number_of_states = len(states)

# initial step
x = np.ndarray(number_of_states, int)
for i in np.arange(0, number_of_states):
    x[i] = (np.argmax(q[i]))

stop_flag = 0
iterations = 0
result = {0: x}
while stop_flag != 3 and iterations < 100:
    # step 1 parameters: states, x
    print(iterations)
    p_matrix = np.ndarray((number_of_states, number_of_states))
    r_matrix = np.ndarray((number_of_states, number_of_states))
    q_cur = np.zeros(number_of_states)
    for i in np.arange(0, number_of_states):
        q_cur[i] = -q[i][x[i]]
        probs, costs = states[i]
        p_matrix[i] = probs[x[i]]
        p_matrix[i][number_of_states - 1] = -1
        p_matrix[i][i] -= 1
    # print(p_matrix)
    v = np.linalg.solve(p_matrix, q_cur)
    g = v[number_of_states - 1]
    v[number_of_states - 1] = 0
    # print(v)

    # step 2  //q_i_k + sum(p_i_j*v_j)
    new_x = np.ndarray(number_of_states, int)
    for i in range(0, number_of_states):
        p, c = states[i]
        number_of_strategies = len(p)
        res = np.ndarray(number_of_strategies)
        for k in range(0, number_of_strategies):
            res[k] = q[i][k] + np.sum(v * p[k])
        new_x[i] = res.argmax()

    if np.array_equal(x, new_x):
        stop_flag += 1
    iterations += 1
    result[iterations] = new_x
    x = new_x

print(result)

from src.utils import get_second_sample
import numpy as np
from src.utils import get_q

debug_enabled = True


def debug(*objects):
    if debug_enabled is True:
        res = ''
        for obj in objects:
            res += str(obj) + " "
        print("[DEBUG] " + res)


# prepare data
states = get_second_sample()
x = np.asarray([0, 1, 2], int)
eps = 0.00000005
number_of_states = len(states)
q = get_q(states)
behavior = []
incomes = []
stop_flag = 0

while stop_flag < 3:
    # step 1
    # 1.  0    = sum(j=1..N)[p(i,j)*g(j)] - g(i),        i = 1..N
    # 2. -q(i) = sum(j=1..N)[p(i,j)*v(j)] - g(i) - v(i), i = 1..N
    #
    #

    p = np.zeros(shape=(number_of_states, number_of_states))
    for i in range(0, number_of_states):
        p[i] = states[i][0][x[i]]
    debug("matrix p:\n", p)

    rank = np.linalg.matrix_rank(p)
    debug("rank p:", rank)
    ergodic_number = number_of_states - rank + 1

    matrix = np.zeros(shape=(number_of_states * 2 - ergodic_number, number_of_states * 2 - ergodic_number))
    for i in range(0, number_of_states):
        for j in range(0, number_of_states):
            matrix[i][j] = p[i][j]

    # вычитаю в результирующей матрице g(i) из 1. и 2.
    # а так же v(i) из 2.
    for i in range(0, number_of_states):
        matrix[i][i] -= 1
    for i in range(number_of_states, number_of_states - ergodic_number):
        matrix[i][i - number_of_states] -= 1
        matrix[i][i] -= 1

    for i in range(number_of_states, number_of_states * 2 - ergodic_number):
        for j in range(number_of_states, number_of_states * 2 - ergodic_number):
            matrix[i][j] += p[i-number_of_states][j-number_of_states]

    right = np.zeros(number_of_states * 2 - ergodic_number)
    for i in range(number_of_states, len(right)):
        right[i] = - q[i - number_of_states][x[i-number_of_states]]

    # g1..gN, v1..vN -->

    solved = np.linalg.solve(matrix, right)
    g = np.asarray(solved[0: number_of_states - 1])
    v = np.zeros(number_of_states)
    v[0: number_of_states - ergodic_number - 1] = solved[number_of_states::1]

    # step 2
    x_new = np.ndarray(shape=number_of_states, dtype=int)
    d = np.ndarray(shape=number_of_states)
    for i in range(0, number_of_states):
        number_of_strategies = len(states[i][0])
        potential_incomes = np.ndarray(number_of_strategies)

        # first criteria
        # max(k)[sum(j=1..N)[p(i)(j)(k)*g(j)]]
        for k in range(0, number_of_strategies):
            potential_incomes[k] = states[i][0][k] * g
        income = max(potential_incomes)
        strategy = potential_incomes.argmax()
        need_use_2_criteria = False
        for k in range(0, number_of_strategies):
            if abs(potential_incomes[k] - income)<eps and k != strategy:
                need_use_2_criteria = True
        if need_use_2_criteria:
            # second criteria:
            # max(k){q(i)(k) + sum(j=1..N)[p(i)(j)(k)*v(j)]}
            for k in range(0, number_of_strategies):
                potential_incomes[k] = q[i][k] + states[i][0][k] * v
            strategy = potential_incomes.argmax()
            income = max(potential_incomes)
        d[i] = income
        x_new[i] = strategy
    behavior.append(x_new)
    incomes.append(d)
    if np.array_equal(x, x_new):
        stop_flag += 1
    else:
        stop_flag = 0

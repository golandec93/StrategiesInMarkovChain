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
x = np.asarray([1, 2, 3], int)
number_of_states = len(states)
q = get_q(states)

# step 1
# 1.  0    = sum(j=1..N)[p(i,j)*g(j)] - g(i),        i = 1..N
# 2. -q(i) = sum(j=1..N)[p(i,j)*v(j)] - g(i) - v(i), i = 1..N
#
#

p = np.zeros(shape=(number_of_states, number_of_states))
for i in range(0, number_of_states):
    for j in range(0, number_of_states):
        p[i][j] = states[i][x[i]][j]
debug("matrix p:\n", p)

rank = np.linalg.matrix_rank(p)
debug("rank p:", rank)
ergodic_number = number_of_states - rank

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
        matrix[i][j] += p[i][j]

right = np.zeros(number_of_states * 2 - ergodic_number)
for i in range(number_of_states, len(right)):
    right[i] = - q[i - number_of_states][x[i]]

# g1..gN, v1..vN -->

solved = np.linalg.solve(matrix, right)
g = np.asarray(solved[0: number_of_states - 1])
v = np.zeros(number_of_states)
v[0: number_of_states - ergodic_number - 1] = solved[number_of_states::1]

# step 2

#

# first criteria

# second criteria

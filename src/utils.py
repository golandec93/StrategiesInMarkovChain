import numpy as np


def get_strategy(length):
    """

    :param length: length of output arrays
    :return: probabilities, costs - 2 numpy's ndarrays
    """
    probabilities = np.random.sample(length)
    probabilities /= np.sum(probabilities)
    costs = (50 * np.random.random() - 30 * np.random.random()) * np.random.sample(length)
    costs -= np.average(costs) * (np.random.random() ** 2)
    return np.asarray(probabilities), np.asarray(costs)


def generate_states(number_of_states, strategies_max=5):
    """

    :param number_of_states:
    :param strategies_max:
    :return: dictionary with keys in range(0, number_of_states) and pairs (strategies~[], costs~[]) as values
    """
    states = {}
    for j in range(0, number_of_states):
        n = int(2 + np.random.random() * (strategies_max - 2))  # number of strategies
        strategies, costs = [], []
        for i in np.arange(0, n):
            strategy, cost = get_strategy(number_of_states)
            strategies.append(strategy)
            costs.append(cost)
        states[j] = strategies, costs
    return states


def states_to_string(states):
    number_of_states = len(states)
    string = ''
    for i in np.arange(0, number_of_states):
        strategies, costs = states[i]
        string += 'state {0} :\n'.format(i + 1)
        for j in np.arange(0, len(strategies)):
            string += '    {0} //// {1}\n'.format(str(strategies[j]), str(costs[j]))
    return string


def print_states(states):
    print(states_to_string(states))


def get_sample():
    return {
        0: ([np.asarray([0.5, 0.5]),
             np.asarray([0.8, 0.2])
             ],
            [np.asarray([9, 3]),
             np.asarray([4, 4])
             ]
            ),

        1: ([np.asarray([0.4, 0.6]),
             np.asarray([0.7, 0.3])
             ],
            [np.asarray([3, -7]),
             np.asarray([1, -19])
             ]
            )
    }


def get_second_sample():
    return {
        0: ([np.asarray([1, 0, 0]),
             np.asarray([0, 1, 0]),
             np.asarray([0, 0, 1])
             ],
            [np.asarray([1, 0, 0]),
             np.asarray([0, 2, 0]),
             np.asarray([0, 0, 3])
             ]
            ),

        1: ([np.asarray([1, 0, 0]),
             np.asarray([0, 1, 0]),
             np.asarray([0, 0, 1])
             ],
            [np.asarray([6, 0, 0]),
             np.asarray([0, 4, 0]),
             np.asarray([0, 0, 5])
             ]
            ),
        2: ([np.asarray([1, 0, 0]),
             np.asarray([0, 1, 0]),
             np.asarray([0, 0, 1])
             ],
            [np.asarray([8, 0, 0]),
             np.asarray([0, 9, 0]),
             np.asarray([0, 0, 7])
             ]
            )
    }


def get_q(states):
    q = {}
    number_of_states = len(states)
    for i in range(0, number_of_states):
        probs, costs = states[i]
        number_of_strategies = len(probs)
        qq = np.ndarray(number_of_strategies)
        for j in np.arange(0, number_of_strategies):
            qq[j] = (np.sum(probs[j] * costs[j]))
        q[i] = qq
    return q

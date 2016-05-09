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
    return probabilities, costs


def generate_states(number_of_states):
    strategies_max = 5

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


def print_states(states):
    number_of_states = len(states)
    for i in np.arange(0, number_of_states):
        strategies, costs = states[i]
        print('state {0} :'.format(i + 1))
        for j in np.arange(0, len(strategies)):
            print('    {0} //// {1}'.format(str(strategies[j]), str(costs[j])))

import string
from src.utils import input_states
from src.utils import get_sample
from src.utils import get_sample
from src.utils import print_states
import src.condecs as condecs
import src.iteralg as iteralg


def execute():
    stop_flag = True
    states = 0
    while(stop_flag):
        print("1. Input Markov Chain")
        print("2. Use sample")
        print("3. Perform algorithm")
        print("0. Exit")
        print('\n\n')
        choice = int(input("You choice: "))
        print('\n\n')
        if choice == 1:
            states = input_states()
        elif choice == 2:
            states = get_sample()
        elif choice == 3:
            print("1. first algorithm")
            print("2. second algorithm")
            choice2 = int(input("You choice: "))
            if choice2 == 1:
                number_of_iterations = int(input("Input number of iterations: "))
                behavior, costs = condecs.eval_behavior(states, number_of_iterations)
            if choice2 == 2:
                behavior, costs = iteralg.eval_behavior(states)
        print('\n\n')
        if choice == 0:
            stop_flag = False



from src.utils import get_sample
from src.utils import print_states
import src.condecs as condecs

states = get_sample()
print_states(states)
behavior, income = condecs.eval_behavior(states, debug_enabled=True)


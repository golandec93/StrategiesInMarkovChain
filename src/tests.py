from src.utils import get_sample
from src.utils import print_states
import src.condecs as condecs
import src.iteralg as iteralg

states = get_sample()
print_states(states)
debug_enabled = True
behavior1, income1 = condecs.eval_behavior(states, debug_enabled=debug_enabled)
behavior2, income2 = iteralg.eval_behavior(states, debug_enabled=debug_enabled)

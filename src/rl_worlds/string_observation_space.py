import random
from gymnasium.spaces import Space
from typing import List


class StringObservationSpace(Space):
    def __init__(self, non_terminal_states: List, terminal_states: List):
        super().__init__((), str)
        self.non_terminal_states = non_terminal_states
        self.terminal_states = terminal_states
        self.all_states = set(self.non_terminal_states + self.terminal_states)

    def contains(self, x):
        return x in self.all_states

    def sample(self):
        return random.choice(self.non_terminal_states)
import gymnasium as gym
from gymnasium.spaces import Discrete

from rl_worlds.string_observation_space import StringObservationSpace


class TunnelWorld(gym.Env):
    """
    A simple tunnel world environment for reinforcement learning.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        max_steps: int = 50,
        **kwargs,
    ):
        super(TunnelWorld, self).__init__()
        self.states = ["A", "B", "C", "D"]
        self.start_state = "B"
        self.terminal_states = ["A", "D"]

        self.action_space = Discrete(2)  # 0: left, 1: right
        self.observation_space = StringObservationSpace(non_terminal_states=["B", "C"], terminal_states=["A", "D"])

        self.reward_map = {
            ("A", 0): 0,
            ("A", 1): 0,
            ("B", 0): 100,
            ("B", 1): 50,
            ("C", 0): -100,
            ("C", 1): -50,
            ("D", 0): 0,
            ("D", 1): 0,
        }

        self.max_steps = max_steps

        self.is_terminated = False
        self.is_truncated = False


    def _get_transition_probability(self, state, action):
        """Get the transition probability for a given state and action."""
        if state not in self.states:
            return 0.0

        if action == 0:  # Move left
            next_state = self.states[self.states.index(state) - 1]
        elif action == 1:  # Move right
            next_state = self.states[self.states.index(state) + 1]
        else:
            return 0.0

        if next_state in self.terminal_states:
            return 1.0
        return 0.0

    def reset(self, seed=None, options=None): # type: ignore
        """Reset the environment to the starting state."""
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.steps = 0
        self.is_terminated = False
        self.is_truncated = False
        return self.current_state, {}

 

    def step(self, action):
        """Execute an action and return next state, reward, terminated, truncated, and info."""

        if self.is_terminated or self.is_truncated:
            return self.current_state, 0, self.is_terminated, self.is_truncated, {}

        if self.steps >= self.max_steps:
            self.is_truncated = True
            return self.current_state, 0, self.is_terminated, self.is_truncated, {}

        self.steps += 1

        current_state_index = self.states.index(self.current_state)

        next_state_index = current_state_index - 1 if action == 0 else current_state_index + 1

        if next_state_index == 0 or next_state_index == len(self.states)-1:
            self.is_terminated = True

        next_state = self.states[next_state_index]
        reward = self.reward_map.get((self.current_state, action), 0)

        self.current_state = next_state

        return self.current_state, reward, self.is_terminated, self.is_truncated, {}

    def render(self):
        """Render the current state of the environment."""

        tunnel_len = len(self.states)
        current_index = self.states.index(self.current_state)

        tunnel = ["-"] * tunnel_len
        tunnel[current_index] = "A"

        print(tunnel)

if __name__ == "__main__":
    env = TunnelWorld()
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated
    print("Episode finished.")
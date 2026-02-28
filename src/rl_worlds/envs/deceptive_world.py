import gymnasium as gym
from gymnasium.spaces import Discrete


class DeceptiveWorldEnv(gym.Env):
    """
    DeceptiveWorld: A grid environment illustrating how greedy agents fail.

    A tempting "direct" path leads to a pseudo target with high per-step rewards
    but suboptimal total return, while the harder path to the real target yields
    low per-step rewards but the highest terminal reward and best cumulative return.

    Grid layout (4×4 example):
     0  1  2  3    <- start at 0 (top-left)
     4  5  6  7
     8  9 10 11
    12 13 14 15    <- pseudo target: 12 (bottom-left), real target: 15 (bottom-right)

    Reward structure:
    - Step into leftmost column state (not start, not pseudo target), agent moved: +10.0
    - Reach pseudo target (bottom-left): +50.0 (terminal)
    - Reach real target (bottom-right): +100.0 (terminal)
    - All other steps: 0.0
    """

    SIZE_MAP = {"small": (4, 4), "medium": (5, 5), "large": (6, 6)}
    metadata = {"render_modes": ["ascii", "human"], "render_fps": 4}

    def __init__(self, size="small", max_steps=100, render_mode=None):
        super().__init__()

        if size not in self.SIZE_MAP:
            raise ValueError(f"size must be one of {list(self.SIZE_MAP.keys())}, got {size!r}")

        rows, cols = self.SIZE_MAP[size]
        self.rows = rows
        self.cols = cols
        n_states = rows * cols

        self.start_state = 0                        # top-left
        self.pseudo_target = (rows - 1) * cols      # bottom-left
        self.real_target = n_states - 1             # bottom-right
        self.leftmost_column = {r * cols for r in range(rows)}

        self.observation_space = Discrete(n_states)
        self.action_space = Discrete(4)  # 0=up, 1=right, 2=down, 3=left

        self.max_steps = max_steps
        self.render_mode = render_mode

        self._state = self.start_state
        self._steps = 0
        self._trajectory = []
        self._terminated = False
        self._truncated = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.start_state
        self._steps = 0
        self._trajectory = []
        self._terminated = False
        self._truncated = False
        return self._state, {"trajectory": list(self._trajectory)}

    def step(self, action):
        if self._terminated or self._truncated:
            return self._state, 0.0, self._terminated, self._truncated, {"trajectory": list(self._trajectory)}

        prev_state = self._state
        row, col = divmod(self._state, self.cols)

        if action == 0:    # up
            next_row, next_col = row - 1, col
        elif action == 1:  # right
            next_row, next_col = row, col + 1
        elif action == 2:  # down
            next_row, next_col = row + 1, col
        elif action == 3:  # left
            next_row, next_col = row, col - 1
        else:
            raise ValueError(f"Invalid action {action}")

        # Clip to grid (stay in place if out of bounds)
        next_row = max(0, min(self.rows - 1, next_row))
        next_col = max(0, min(self.cols - 1, next_col))
        next_state = next_row * self.cols + next_col

        actually_moved = next_state != prev_state
        self._state = next_state
        self._steps += 1

        if next_state == self.real_target:
            reward = 100.0
            self._terminated = True
        elif next_state == self.pseudo_target:
            reward = 50.0
            self._terminated = True
        elif (
            next_state in self.leftmost_column
            and next_state != self.start_state
            and actually_moved
        ):
            reward = 10.0
        else:
            reward = 0.0

        if not self._terminated and self._steps >= self.max_steps:
            self._truncated = True

        self._trajectory.append((prev_state, action))

        return self._state, reward, self._terminated, self._truncated, {"trajectory": list(self._trajectory)}

    def render(self):
        separator = "+" + "+".join(["----"] * self.cols) + "+"
        lines = []

        for r in range(self.rows):
            lines.append(separator)
            row_cells = []
            for c in range(self.cols):
                state = r * self.cols + c
                cell = "  X " if state == self._state else f"{state:3} "
                row_cells.append(cell)
            lines.append("|" + "|".join(row_cells) + "|")

        lines.append(separator)
        render_str = "\n".join(lines)

        if self.render_mode == "human":
            print(render_str)
            return None
        elif self.render_mode == "ascii":
            return render_str

from gymnasium.utils.env_checker import check_env
from rl_worlds.envs.grid_world import GridWorldEnv
from rl_worlds.envs.random_walk import RandomWalkEnv
from rl_worlds.envs.windy_grid_world import WindyGridWorldEnv

check_env(RandomWalkEnv())
check_env(GridWorldEnv())
check_env(WindyGridWorldEnv())

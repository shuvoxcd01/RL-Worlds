from gymnasium.utils.env_checker import check_env
from rl_worlds.envs.grid_world import GridWorldEnv
from rl_worlds.envs.random_walk import RandomWalkEnv
from rl_worlds.envs.windy_grid_world import WindyGridWorldEnv
from rl_worlds.envs.tunnel_world import TunnelWorld
import gymnasium as gym


env = gym.make("rl_worlds/RandomWalk-v0", num_states=6)
# env = gym.make("rl_worlds/ThousandStatesRandomWalk-v0")
# env = gym.make("rl_worlds/GridWorld-v0")
# env = gym.make("rl_worlds/WindyGridWorld-v0")
# env = gym.make("rl_worlds/ShortCorridor-v0")
# env = gym.make("rl_worlds/TunnelWorld-v0")
check_env(env.unwrapped)


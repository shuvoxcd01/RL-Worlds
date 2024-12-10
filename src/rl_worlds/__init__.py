from gymnasium.envs.registration import register

register(
    id="rl_worlds/RandomWalk-v0",
    entry_point="rl_worlds.envs:RandomWalkEnv",
)

register(
    id="rl_worlds/ThousandStatesRandomWalk-v0",
    entry_point="rl_worlds.envs:ThousandStatesRandomWalkEnv",
)

register(id="rl_worlds/GridWorld-v0", entry_point="rl_worlds.envs:GridWorldEnv")

register(
    id="rl_worlds/WindyGridWorld-v0", entry_point="rl_worlds.envs:WindyGridWorldEnv"
)

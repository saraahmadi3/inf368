from gymnasium.envs.registration import register

register(
    id='mountain/GridWorld-v0',
    entry_point="mountain.envs:MountainEnv",
)
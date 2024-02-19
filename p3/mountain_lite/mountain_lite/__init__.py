from gymnasium.envs.registration import register

register(
    id='mountain_lite/GridWorld-v0',
    entry_point="mountain_lite.envs:MountainLiteEnv",
)
from gymnasium.envs.registration import register

config_v1 = {"img": "cookie", "width" : 10, "lifetime":5, "friction": lambda vel: - abs(vel)*vel * 0.05}
config_v2 = {"img": "cake", "width" : 5, "lifetime":4, "friction": lambda vel: - abs(vel)*vel * 0.08}
config_v3 = {"img": "scone", "width" : 12, "lifetime":7, "friction": lambda vel: - abs(vel)*vel * 0.1}


register(
    id='cookiedisaster-v1',
    entry_point="cookiedisaster.envs:CookieDisasterEnv",
    kwargs={'config': config_v1}
)

register(
    id='cookiedisaster-v2',
    entry_point="cookiedisaster.envs:CookieDisasterEnv",
    kwargs={'config': config_v2}
)

register(
    id='cookiedisaster-v3',
    entry_point="cookiedisaster.envs:CookieDisasterEnv",
    kwargs={'config': config_v3}
)
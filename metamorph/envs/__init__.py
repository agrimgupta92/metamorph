from gym.envs.registration import register

register(
    id="Unimal-v0",
    entry_point="metamorph.envs.tasks.task:make_env",
    max_episode_steps=1000,
)

register(
    id="GeneralWalker2D-v0",
    entry_point="metamorph.envs.tasks.gen_walker_2d:make_env",
    max_episode_steps=1000,
)

CUSTOM_ENVS = ["Unimal-v0", "GeneralWalker2D-v0"]
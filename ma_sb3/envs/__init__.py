from gymnasium.envs.registration import register

from ma_sb3.envs.pretator_prey_v0 import PredatorPreyEnv

register(
    id='PredatorPrey-v0',
    entry_point='ma_sb3.envs.predator_prey_v0.PredatorPreyEnv',
    max_episode_steps=100,
)

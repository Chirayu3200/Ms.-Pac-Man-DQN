import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import config

gym.register_envs(ale_py)   # makes the "ALE/..." games available to gymnasium


def make_env(render_mode=None):
    # frameskip=1 because AtariPreprocessing does the frame skipping itself
    env = gym.make(config.ENV_ID, frameskip=1, render_mode=render_mode)

    # Shrink the screen to 84x84 grayscale, repeat each action for 4 frames,
    # and do a random number of "do nothing" moves at the start of each game.
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True
    )

    # Give the AI the last 4 frames at once, so it can see movement.
    env = FrameStackObservation(env, stack_size=4)
    return env
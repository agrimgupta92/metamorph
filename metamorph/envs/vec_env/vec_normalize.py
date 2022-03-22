import gym
import numpy as np

from .running_mean_std import RunningMeanStd
from .vec_env import VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        venv,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        training=True,
        obs_to_norm=None
    ):
        VecEnvWrapper.__init__(self, venv)

        self.ob_rms = self._init_ob_rms(ob, obs_to_norm)
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

    def _init_ob_rms(self, ob, obs_to_norm):
        if not ob:
            return None

        obs_space = self.observation_space
        ob_rms = {}

        if isinstance(obs_space, gym.spaces.Dict):
            for obs_type in obs_to_norm:
                shape = obs_space[obs_type].shape
                ob_rms[obs_type] = RunningMeanStd(shape=shape)
        else:
            shape = obs_space.shape
            ob_rms["proprioceptive"] = RunningMeanStd(shape=shape)

        return ob_rms

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        self.ret[news] = 0.0
        return obs, rews, news, infos

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            for obs_type in self.ob_rms.keys():
                obs = self._obfilt_helper(obs, obs_type)
            return obs
        else:
            return obs

    def _obfilt_helper(self, obs, obs_type, update=True):
        if isinstance(obs, dict):
            obs_p = obs[obs_type]
        else:
            obs_p = obs

        if self.training and update:
            self.ob_rms[obs_type].update(obs_p)

        obs_p = np.clip(
            (obs_p - self.ob_rms[obs_type].mean)
            / np.sqrt(self.ob_rms[obs_type].var + self.epsilon),
            -self.clipob,
            self.clipob,
        )
        if isinstance(obs, dict):
            obs[obs_type] = obs_p
        else:
            obs = obs_p
        return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

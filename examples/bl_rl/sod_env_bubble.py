#!/usr/bin/env python3

import numpy as np

from smartsim.log import get_logger
from smartsod2d.utils import numpy_str

from smartsod2d.sod_env_base import SodEnv

logger = get_logger(__name__)

class SodEnvBubble(SodEnvBase):
    """
    SOD2D environment(s) extending the tf_agents.environments.PyEnvironment class.
    Inherits from the SodEnvBase environment, the functions 
    defined here are specific for the reduction of the 
    recirculation bubble in a turbulent boundary layer
    """
    def _redistribute_state(self):
        """
        Redistribute state across MARL pseudo-environments.
        Make sure the witness points are written in such that the first moving coordinate is x, then y, and last z.

        The redistributed state is saved in variable self._state_marl
        """
        state_extended = np.concatenate((self._state, self._state, self._state), axis=1)
        plane_wit = self.witness_xyz[0] * self.witness_xyz[1]
        block_wit = int(plane_wit * (self.witness_xyz[2] / self.marl_n_envs))
        for i in range(self.cfd_n_envs):
            for j in range(self.marl_n_envs):
                self._state_marl[i * self.marl_n_envs + j,:] = state_extended[i, block_wit * (j - self.marl_neighbors) + \
                    self.n_state:block_wit * (j + self.marl_neighbors + 1) + self.n_state]


    def _get_reward(self):
        """
        Obtain the local unprocessed value of the reward from each CFD environment and compute the local/global reward for the problem at hand
        """
        for i in range(self.cfd_n_envs):
            if self._step_type[i] > 0: # environment still running
                self.client.poll_tensor(self.reward_key[i], 100, self.poll_time)
                try:
                    reward = self.client.get_tensor(self.reward_key[i])
                    self.client.delete_tensor(self.reward_key[i])
                    logger.debug(f"[Env {i}] Got reward: {numpy_str(reward)}")

                    local_reward = -reward / self.reward_norm
                    self._local_reward[i, :] = local_reward
                    global_reward = np.mean(local_reward)
                    for j in range(self.marl_n_envs):
                        self._reward[i * self.marl_n_envs + j] = self.reward_beta * global_reward + (1.0 - self.reward_beta) * local_reward[j]
                except Exception as exc:
                    raise Warning(f"Could not read reward from key: {self.reward_key[i]}") from exc

    def _set_action(self, action):
        """
        Write actions for each environment to be polled by the corresponding Sod2D environment.
        Action clipping must be performed within the environment: https://github.com/tensorflow/agents/issues/216 when using PPO
        """
        # scale actions and reshape for SOD2D
        action = action * self.action_bounds[1] if self.mode == "collect" else action
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        for i in range(self.cfd_n_envs):
            for j in range(self.marl_n_envs):
                self._action[i, j] = action[i * self.marl_n_envs + j]
        # apply zero-net-mass-flow strategy
        self._action_znmf = np.repeat(self._action, 2, axis=-1)
        self._action_znmf[..., 1::2] *= -1.0
        # write action into database
        for i in range(self.cfd_n_envs):
            self.client.put_tensor(self.action_key[i], self._action_znmf[i, ...].astype(self.sod_dtype))
                # np.zeros(self.n_action * 2, dtype=self.sod_dtype))
            logger.debug(f"[Env {i}] Writing (half) action: {numpy_str(self._action[i, :])}")

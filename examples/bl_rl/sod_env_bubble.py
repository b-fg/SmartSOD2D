#!/usr/bin/env python3

import numpy as np

from smartsim.log import get_logger
from smartsod2d.utils import numpy_str

from smartsod2d.sod_env_base import SodEnvBase

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

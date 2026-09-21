"""SB3 VecEnv around WarpOuterBackend (N envs, case list sampled per reset)."""
import copy
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvObs
from warp_backend.backend import WarpOuterBackend


class WarpOuterVecEnv(VecEnv):
    def __init__(self, plant, inner_study, inner_model, case, n_envs=8, seed=0,
                 reward_shape="l2", failure=-1040.0, effort_scale=1.0, cases=None,
                 memory_divisor=0.5, memory_cost=0.5, smooth_alpha=1.0, quad_weight=100.0,
                 backend="numpy"):
        """backend="numpy" preserves the reference rollout (fyp runner default).
        backend="device" uses the device-resident loop (same contract)."""
        self.cases = [copy.deepcopy(c) for c in (cases or [case])]
        self.rng = np.random.default_rng(seed)
        if backend == "device":
            from warp_backend.rollout import FastOuterBackend
            self.be = FastOuterBackend(plant, inner_study, inner_model, self.cases, seed=seed,
                                       reward_shape=reward_shape, failure=failure,
                                       effort_scale=effort_scale, memory_divisor=memory_divisor,
                                       memory_cost=memory_cost, smooth_alpha=smooth_alpha,
                                       quad_weight=quad_weight)
        elif backend == "numpy":
            self.be = WarpOuterBackend(plant, inner_study, inner_model, self.cases, seed=seed,
                                       reward_shape=reward_shape, failure=failure,
                                       effort_scale=effort_scale, memory_divisor=memory_divisor,
                                       memory_cost=memory_cost, smooth_alpha=smooth_alpha,
                                       quad_weight=quad_weight)
        else:
            raise ValueError(f"Unknown rollout backend {backend!r}")
        self.case = self.cases[int(self.rng.integers(len(self.cases)))]
        obs, _ = self.be.reset(n_envs, case=self.case, seed=seed)
        super().__init__(
            num_envs=n_envs,
            observation_space=Box(-1, 1, shape=(7,), dtype=np.float32),
            action_space=Box(-1, 1, shape=(1,), dtype=np.float32),
        )
        self._actions = None

    def reset(self) -> VecEnvObs:
        self.case = self.cases[int(self.rng.integers(len(self.cases)))]
        obs, _ = self.be.reset(self.num_envs, case=self.case)
        return obs.astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=float).reshape(self.num_envs, 1)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, terms, truncs, infos = self.be.step(self._actions)
        dones = terms | truncs
        obs = obs.astype(np.float32)
        for j in np.where(dones)[0]:
            # SB3 timeout key (was TimeLimit_truncated, which disabled bootstrapping).
            infos[j] = dict(infos[j], terminal_observation=obs[j].copy(),
                            **{"TimeLimit.truncated": bool(truncs[j] and not terms[j])})
            obs[j] = self.be.reset_env(int(j))
        return obs, rews.astype(float), dones, infos

    def seed(self, seed=None):
        return [seed]

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, *args, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, *args, **kwargs):
        return [False] * self.num_envs

    def get_images(self):
        raise NotImplementedError

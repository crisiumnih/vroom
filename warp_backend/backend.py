"""Warp outer-loop rollout backend (phase 2b).

N parallel outer envs: numpy per-env CurrentEncoders + ONE batched torch
inner-actor forward on CUDA + Warp 100us physics. Outer arithmetic
(reward terms, trip>4A, memory z, truncation) mirrors
`tools/outer_rl/environment.py` exactly. GEM imports are read-only.
"""
import copy
import numpy as np
import torch
import warp as wp

import sys
from pathlib import Path
import os
_here = Path(__file__).resolve()
def _gem_root():
    sub = _here.parents[1] / "third_party" / "gym-electric-motor"
    if (sub / "benchmarks" / "bldc" / "current_rl" / "core.py").exists():
        return str(sub)
    env = os.environ.get("VROOM_GEM_PATH")
    if env and Path(env).exists():
        return env
    return "/home/sra/prajwal/fyp/gym-electric-motor"
GEM_ROOT = _gem_root()
if GEM_ROOT not in sys.path:
    sys.path.insert(0, GEM_ROOT)

from benchmarks.bldc.environment import schedule_value
from benchmarks.bldc.current_rl.core import CurrentEncoder
from warp_backend import fullstep as fs

wp.init()

SQRT3_2 = 0.8660254037844386


def abc_to_dq(i_abc, eps):
    alpha = (2.0 / 3.0) * (i_abc[:, 0] - 0.5 * i_abc[:, 1] - 0.5 * i_abc[:, 2])
    beta = (2.0 / 3.0) * (SQRT3_2 * i_abc[:, 1] - SQRT3_2 * i_abc[:, 2])
    ce, se = np.cos(eps), np.sin(eps)
    return ce * alpha + se * beta, -se * alpha + ce * beta


def bemf_shape(eps):
    TWO_PI = 2 * np.pi
    th = np.mod(eps + 5 * np.pi / 6, TWO_PI)
    def pha(t):
        return np.where(t < TWO_PI / 3, 1.0,
               np.where(t < np.pi, 1.0 - 2.0 * (t - TWO_PI / 3) / (np.pi - TWO_PI / 3),
               np.where(t < 5 * np.pi / 3, -1.0, -1.0 + 2.0 * (t - 5 * np.pi / 3) / (TWO_PI - 5 * np.pi / 3))))
    return pha(th), pha(np.mod(th - TWO_PI / 3, TWO_PI)), pha(np.mod(th - 4 * np.pi / 3, TWO_PI))


class WarpOuterBackend:
    """Batched outer env. States: omega, i_abc, eps per env (float64)."""

    def __init__(self, plant, inner_study, inner_model, scenarios, seed=0,
                 reward_shape="l2", failure=-1040.0, effort_scale=1.0,
                 memory_divisor=0.5, memory_cost=0.5, smooth_alpha=1.0,
                 quad_weight=100.0):
        self.plant = copy.deepcopy(plant)
        self.study = copy.deepcopy(inner_study)
        self.model = inner_model
        self.scenarios = copy.deepcopy(scenarios)
        self.rng = np.random.default_rng(seed)
        assert reward_shape in ("l2", "l1", "qeff")
        self.shape, self.failure = reward_shape, float(failure)
        self.effort = float(effort_scale)
        self.mem_div, self.mem_cost = float(memory_divisor), float(memory_cost)
        self.alpha = float(smooth_alpha)
        self.qw = float(quad_weight)
        assert plant["tau_s"] == 1e-4
        contract = inner_study["contract"]
        self.tau_i, self.tau_o, self.hold = 1e-4, 1e-3, 10
        self.cur_lim = plant["controller"]["current_reference_limit_a"]
        self.n = None
        self.contract = contract

    def reset(self, n, case=None, seed=None):
        self.n = n
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.case = copy.deepcopy(case or self.scenarios[int(self.rng.integers(len(self.scenarios)))])
        self.om = np.zeros(n)
        self.i = np.zeros((n, 3))
        self.eps = np.zeros(n)
        self.enc = [CurrentEncoder(self.contract, "direct") for _ in range(n)]
        for e in self.enc:
            e.reset()
        self.z = np.zeros(n)
        self.prev_cmd = np.zeros(n)
        self.cmd_f = np.zeros(n)
        self.phys = np.zeros(n, dtype=int)
        dur = round(self.case["duration_s"] / self.tau_i)
        self.dur = np.full(n, dur)
        self.done = np.zeros(n, dtype=bool)
        d_s = wp.array(np.zeros((n, 5)), dtype=wp.float64, device="cuda:0")
        self.d_s = d_s
        self.d_a = wp.zeros((n, 2), dtype=wp.float64, device="cuda:0")
        self.d_d = wp.zeros(n, dtype=wp.float64, device="cuda:0")
        self.d_o = wp.zeros((n, 6), dtype=wp.float64, device="cuda:0")
        return self._obs()

    def reset_env(self, j, seed=None):
        """Re-init a single env (for VecEnv auto-reset); layout fixed at reset()."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        self.case = copy.deepcopy(self.case)  # shared case across envs in this backend
        self.om[j], self.i[j], self.eps[j] = 0.0, np.zeros(3), 0.0
        self.enc[j] = CurrentEncoder(self.contract, "direct")
        self.enc[j].reset()
        self.z[j], self.prev_cmd[j] = 0.0, 0.0
        self.cmd_f[j] = 0.0
        self.phys[j] = 0
        self.done[j] = False
        wp.launch(fs.reset_row_kernel, dim=1,
                  inputs=[self.d_s, j], device="cuda:0")
        wp.synchronize()
        return self._single_obs(j)

    def _single_obs(self, j):
        ref = schedule_value(self.case["reference"], int(self.phys[j]), self.tau_i)
        isd, isq = abc_to_dq(self.i[j:j + 1], self.eps[j:j + 1])
        v = np.array([self.om[j] / 25, ref / 25, (ref - self.om[j]) / 25,
                      float(isd[0]) / 4, float(isq[0]) / 4,
                      self.prev_cmd[j] / 1.5, self.z[j]], dtype=np.float32)
        assert np.isfinite(v).all()
        return np.clip(v, -1, 1)

    def _obs(self):
        ref = np.array([schedule_value(self.case["reference"], int(p), self.tau_i) for p in self.phys])
        isd, isq = abc_to_dq(self.i, self.eps)
        v = np.stack([self.om / 25, ref / 25, (ref - self.om) / 25, isd / 4, isq / 4,
                      self.prev_cmd / 1.5, self.z], axis=1).astype(np.float32)
        assert np.isfinite(v).all()
        return np.clip(v, -1, 1), ref

    def step(self, actions):
        """actions: (N,1) in [-1,1]. Returns obs, rewards, term, trunc, infos."""
        a = np.asarray(actions, dtype=float).reshape(self.n, 1)
        cmd = 1.5 * np.clip(a[:, 0], -1, 1)
        # Smoothing (thesis Sec 4.2): plant-facing reference is low-passed;
        # delta penalty and previous_command stay on the raw actor output.
        self.cmd_f = self.alpha * cmd + (1.0 - self.alpha) * self.cmd_f
        cmdf = self.cmd_f
        obs0, ref0 = self._obs()
        om_pre = self.om.copy()
        z0 = self.z.copy()
        delta = np.abs(cmd - self.prev_cmd) / 3.0
        rewards = np.zeros(self.n)
        term = np.zeros(self.n, dtype=bool)
        trunc = np.zeros(self.n, dtype=bool)
        fail = [None] * self.n
        counts = np.zeros(self.n, dtype=int)
        fa, fb, fc = bemf_shape(self.eps)
        tprev = 0.0955 * (fa * self.i[:, 0] + fb * self.i[:, 1] + fc * self.i[:, 2])
        for _ in range(self.hold):
            live = ~(term | trunc) & (self.phys < self.dur)
            if not live.any():
                break
            dist = np.array([schedule_value(self.case["disturbance"], int(p), self.tau_i) for p in self.phys])
            isd, isq = abc_to_dq(self.i, self.eps)
            # batched inner forward
            obatch = np.stack([e.encode(
                {"i_sd": float(d), "i_sq": float(q), "omega": float(o), "epsilon": float(e_)},
                np.array([0.0, float(c)]) * min(1.0, self.cur_lim / max(abs(float(c)), 1e-30)))
                for e, d, q, o, e_, c in zip(self.enc, isd, isq, self.om, self.eps, cmdf)])
            with torch.no_grad():
                raw = self.model.actor.forward(torch.as_tensor(obatch, device="cuda")).cpu().numpy()
            volt = np.zeros((self.n, 2))
            for j in np.where(live)[0]:
                applied, _, _ = self.enc[j].action(raw[j], {"i_sd": float(isd[j]), "i_sq": float(isq[j])})
                volt[j] = applied
            d_a = wp.array(np.ascontiguousarray(volt), dtype=wp.float64, device="cuda:0")
            d_d = wp.array(np.ascontiguousarray(dist), dtype=wp.float64, device="cuda:0")
            wp.launch(fs.fullstep_once, dim=self.n,
                      inputs=[self.d_s, d_a, d_d, wp.float64(self.tau_i), self.d_o], device="cuda:0")
            wp.synchronize()
            row = self.d_o.numpy()
            self.om, self.i, self.eps = row[:, 0].copy(), row[:, 1:4].copy(), row[:, 4].copy()
            tq = row[:, 5]
            self.phys[live] += 1
            counts[live] += 1
            peak = np.abs(self.i).max(axis=1)
            isd2, isq2 = abc_to_dq(self.i, self.eps)
            t_term = peak > 4.0
            term[live] = t_term[live]
            for j in np.where(live)[0]:
                if t_term[j]:
                    fail[j] = "phase_current_trip"
            l2speed = 4.0 * np.clip((ref0[live] - self.om[live]) / 25, -1, 1) ** 2
            if self.shape == "l1":
                en = np.clip((ref0[live] - self.om[live]) / 25, -1, 1)
                l2speed = l2speed + (4.0 * (np.abs(en) - en ** 2))
            elif self.shape == "qeff":
                # Quadratic tracking (thesis Eq.20 style): qw matched so cost
                # at en=0.04 equals the L1 cost there; smooth gradient at zero.
                en = np.clip((ref0[live] - self.om[live]) / 25, -1, 1)
                l2speed = self.qw * en ** 2
            rewards[live] += -(l2speed
                + 0.5 * np.clip(np.hypot(isd2[live], isq2[live]) / 4, 0, 1) ** 2
                + self.effort * 0.1 * np.clip(delta[live], 0, 1) ** 2 + self.mem_cost * z0[live] ** 2
                + self.effort * 0.1 * np.clip(np.abs(tq[live] - tprev[live]) / 0.1, 0, 1) ** 2)
            tprev = tq.copy()
        cnt = np.maximum(counts, 1)
        rewards = np.where(counts > 0, rewards / cnt, 0.0)
        rewards[term] = self.failure
        self.z = np.clip(z0 + counts * self.tau_i * np.clip((ref0 - om_pre) / 25, -1, 1) / self.mem_div, -1, 1)
        self.prev_cmd = cmd
        fin = (~term) & (self.phys >= self.dur)
        trunc[fin] = True
        self.done = term | trunc
        obs, _ = self._obs()
        infos = [{"failure": fail[j], "physics_steps": int(self.phys[j])} for j in range(self.n)]
        return obs, rewards, term, trunc, infos

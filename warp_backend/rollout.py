"""Device-resident inner rollout loop (P1/P2 optimization).

Same controller contract as WarpOuterBackend.step; the host only sees the
SB3 outer-step boundary (actions in; obs/rewards/dones/infos out, one sync).
WarpOuterBackend is the unchanged reference: contract changes must land in
both until the reference is retired. Differences from the reference are
intentional defect repairs, flagged below with [REPAIR].

[REPAIR-1] Terminated lanes freeze at first termination (plant, encoder,
controller memory, terminal snapshot). The reference keeps advancing them.
[REPAIR-2] infos carry `inner_steps` (executed-step deltas). The cumulative
`physics_steps` key is preserved for existing consumers.
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
from warp_backend import fullstep as fs
from warp_backend.backend import WarpOuterBackend
from warp_backend.plant import MotorParams, resolve_plants

wp.init()

SQRT3_2 = 0.8660254037844386


def validate_plant(plant):
    """Structural check: convertible to MotorParams (any physical values).

    The nominal-only restriction is gone (master_plan PR1); per-lane values
    flow through the params table built in reset().
    """
    if isinstance(plant, MotorParams):
        return plant.validated()
    from warp_backend.plant import from_legacy_dict
    return from_legacy_dict(plant)


def build_tables(case, tau_i, dur):
    """Per-step reference/disturbance columns with exact round() semantics.

    schedule_value takes the last event with round(t/dt)<=k; filling segments
    between consecutive integer boundaries reproduces it exactly. Rejects
    unsorted schedules rather than silently reordering them.
    """
    cols = {}
    for key in ("reference", "disturbance"):
        sched = case[key]
        ts = [t for t, _ in sched]
        if any(b < a for a, b in zip(ts, ts[1:])):
            raise ValueError(f"Unsorted {key} schedule")
        bounds = [round(t / tau_i) for t, _ in sched]
        col = np.empty(dur + 1)
        for (b0, (_, v0)), b1 in zip(zip(bounds, sched), bounds[1:] + [dur + 1]):
            col[max(b0, 0):max(b1, 0)] = v0[1] if isinstance(v0, (list, tuple)) else v0
        cols[key] = col
    return cols["reference"], cols["disturbance"]


class FastOuterBackend(WarpOuterBackend):
    """Drop-in device path. API identical to WarpOuterBackend.step/reset."""

    def __init__(self, *args, **kwargs):
        self.control = kwargs.pop('control', 'direct')
        if self.control not in ('direct', 'iasa'):
            raise ValueError(f"Unknown control {self.control}")
        super().__init__(*args, **kwargs)
        if self.control == 'iasa' and abs(float(self.alpha) - 0.5) > 1e-12:
            raise ValueError('IASA contract fixes filter alpha at 0.5')
        if self.shape == 'candidate' and self.control != 'iasa':
            raise ValueError('Candidate reward (v4) requires iasa control')
        if self.control == 'iasa' and self.shape not in ('l1', 'candidate'):
            raise ValueError(f"IASA supports reward shapes l1/candidate, not {self.shape}")
        validate_plant(self.plant)
        c = self.contract
        self.H = int(c["history"])
        self.SCL = float(c["current_scale_a"])
        self.SPD = float(c["speed_scale_rad_s"])
        self.BND = float(c["integrator_bound"])
        self.ANL = float(c["action_norm_limit"])
        self.wstream = wp.Stream("cuda:0")
        handle = self.wstream.cuda_stream
        self.tstream = torch.cuda.ExternalStream(int(handle))
        self._dev = None

    def reset(self, n, case=None, seed=None, plants=None):
        self.n = n
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        import torch
        if self.sense_gen is None:
            self.sense_gen = torch.Generator(device='cuda')
        if seed is not None:
            self.sense_gen.manual_seed(seed)
        self.lane_plants = resolve_plants(plants if plants is not None else self.plant, n)
        if case is None:
            # Independent scenario per lane (training coverage).
            self.case_ids = self.rng.integers(len(self.scenarios), size=n)
        else:
            self.case_ids = np.full(n, -1)
            self.case = copy.deepcopy(case)
        dev = torch.device("cuda:0")
        durs, cols_r, cols_d, names, lane_cases = [], [], [], [], []
        for j in range(n):
            cj = self.case if self.case_ids[j] == -1 else self.scenarios[int(self.case_ids[j])]
            dj = round(cj["duration_s"] / self.tau_i)
            rj, djc = build_tables(cj, self.tau_i, dj)
            durs.append(dj)
            cols_r.append(rj)
            cols_d.append(djc)
            names.append(cj.get("name", f"case{j}"))
            lane_cases.append(copy.deepcopy(cj))
        self.case_names = names
        self.lane_cases = lane_cases
        if self.case_ids[0] != -1:
            self.case = copy.deepcopy(lane_cases[0])
        # Table height covers every configured scenario so auto-resets fit.
        T = max(round(c["duration_s"] / self.tau_i) for c in self.scenarios)
        ref_tab = np.zeros((T + 1, n))
        dist_tab = np.zeros((T + 1, n))
        for j in range(n):
            ref_tab[:durs[j] + 1, j] = cols_r[j]
            dist_tab[:durs[j] + 1, j] = cols_d[j]
        self.dur = np.asarray(durs)
        d = {}
        d["plant"] = torch.zeros((n, 5), dtype=torch.float64, device=dev)
        d["hist"] = torch.zeros((n, self.H, 2), dtype=torch.float32, device=dev)
        d["prev"] = torch.zeros((n, 4), dtype=torch.float32, device=dev)
        d["cmd_f"] = torch.zeros((n,), dtype=torch.float32, device=dev)
        d["z"] = torch.zeros((n,), dtype=torch.float64, device=dev)
        d["pcmd"] = torch.zeros((n,), dtype=torch.float32, device=dev)
        d["bias"] = torch.zeros((n,), dtype=torch.float32, device=dev)
        d["h_spd"] = torch.zeros((n, 20), dtype=torch.float32, device=dev)
        d["h_err"] = torch.zeros((n, 20), dtype=torch.float32, device=dev)
        d["h_cmd"] = torch.zeros((n, 20), dtype=torch.float32, device=dev)
        d["clk"] = torch.zeros((n,), dtype=torch.int64, device=dev)
        d["dur"] = torch.as_tensor(self.dur.copy(), dtype=torch.int64, device=dev)
        d["ar_N"] = torch.arange(n, dtype=torch.int64, device=dev)
        d["ref_tab"] = torch.as_tensor(np.ascontiguousarray(ref_tab), dtype=torch.float64, device=dev)
        d["dist_tab"] = torch.as_tensor(np.ascontiguousarray(dist_tab), dtype=torch.float64, device=dev)
        d["volt"] = torch.zeros((n, 2), dtype=torch.float64, device=dev)
        d["dist_b"] = torch.zeros((n,), dtype=torch.float64, device=dev)
        d["out"] = torch.zeros((n, 6), dtype=torch.float64, device=dev)
        d["live"] = torch.zeros((n,), dtype=torch.int32, device=dev)
        d["done"] = torch.zeros((n,), dtype=torch.bool, device=dev)
        d["snap"] = torch.zeros((n, 5), dtype=torch.float64, device=dev)
        d["fail"] = torch.zeros((n,), dtype=torch.int8, device=dev)
        d["rew"] = torch.zeros((n,), dtype=torch.float64, device=dev)
        d["cnt"] = torch.zeros((n,), dtype=torch.int64, device=dev)
        d["tprev"] = torch.zeros((n,), dtype=torch.float64, device=dev)
        d["phys_cum"] = np.zeros(n, dtype=int)
        d["sense"] = torch.zeros((n, 5), dtype=torch.float64, device=dev)
        d["ifilt"] = torch.zeros((n, 2), dtype=torch.float64, device=dev)
        d["eps_hist"] = torch.zeros((n, 11), dtype=torch.float64, device=dev)
        d["volt_prev"] = torch.zeros((n, 2), dtype=torch.float64, device=dev)
        d["params"] = torch.as_tensor(np.ascontiguousarray([p.row() for p in self.lane_plants]),
                                      dtype=torch.float64, device=dev)
        d["ke"] = d["params"][:, 3].clone()
        if self.control == 'iasa':
            # Initialize lag histories with the initial measured frame (raw units).
            r0 = ref_tab[0]
            d["h_err"][:] = torch.as_tensor(r0, dtype=torch.float32, device=dev).unsqueeze(1)
        # Warp views share storage; keep both refs alive for the layout lifetime.
        d["w_plant"] = wp.from_torch(d["plant"])
        d["w_volt"] = wp.from_torch(d["volt"])
        d["w_dist"] = wp.from_torch(d["dist_b"])
        d["w_out"] = wp.from_torch(d["out"])
        d["w_live"] = wp.from_torch(d["live"])
        d["w_params"] = wp.from_torch(d["params"])
        self._dev = d
        # Host mirrors kept only for infos/diagnostics, not for stepping.
        self.phys = np.zeros(n, dtype=int)
        self.done = np.zeros(n, dtype=bool)
        with torch.cuda.stream(self.tstream):
            torch.cuda.synchronize()
        return self._read_obs()

    def reset_env(self, j, seed=None):
        d = self._dev
        # Independent resample per lane (training coverage, not shared case).
        if seed is not None:
            lane_rng = np.random.default_rng(seed)
            idx = int(lane_rng.integers(len(self.scenarios)))
        else:
            idx = int(self.rng.integers(len(self.scenarios)))
        cj = copy.deepcopy(self.scenarios[idx])
        dj = round(cj["duration_s"] / self.tau_i)
        assert dj + 1 <= d["ref_tab"].shape[0], "scenario longer than table"
        rj, djc = build_tables(cj, self.tau_i, dj)
        d["ref_tab"][:dj + 1, j] = torch.as_tensor(rj, dtype=torch.float64, device=d["ref_tab"].device)
        d["dist_tab"][:dj + 1, j] = torch.as_tensor(djc, dtype=torch.float64, device=d["dist_tab"].device)
        d["dur"][j] = dj
        self.dur[j] = dj
        self.case_ids[j] = idx
        self.case_names[j] = cj.get("name", f"case{j}")
        self.lane_cases[j] = cj
        d["plant"][j].zero_()
        d["sense"][j].zero_()
        d["ifilt"][j].zero_()
        d["eps_hist"][j].zero_()
        d["volt_prev"][j].zero_()
        d["hist"][j].zero_()
        d["prev"][j].zero_()
        d["cmd_f"][j] = 0.0
        d["z"][j] = 0.0
        d["pcmd"][j] = 0.0
        d["bias"][j] = 0.0
        d["h_spd"][j].zero_()
        d["h_err"][j].zero_()
        d["h_cmd"][j].zero_()
        if self.control == 'iasa':
            d["h_err"][j] = float(rj[0])
        d["clk"][j] = 0
        d["done"][j] = False
        d["fail"][j] = 0
        d["snap"][j].zero_()
        self.phys[j] = 0
        self.done[j] = False
        # Lane resampled to an independent scenario (coverage).
        return self._single_host_obs(j)

    def _single_host_obs(self, j):
        cj = self.lane_cases[j] if hasattr(self, 'lane_cases') else self.case
        ref = float(schedule_value(cj["reference"], 0, self.tau_i))
        if self.control == 'iasa':
            o = np.zeros(19, dtype=np.float32)
            o[1] = ref / 25
            o[2] = ref / 25
            o[3] = np.tanh(ref / 0.05)
            o[9] = 1.0
            o[11] = o[14] = o[17] = ref / 25
            return np.clip(o, -1, 1)
        return np.clip(np.array([0.0, ref / 25, ref / 25, 0.0, 0.0, 0.0, 0.0],
                                dtype=np.float32), -1, 1)

    def _read_obs(self):
        with torch.no_grad():
            o = self._assemble_iasa_obs() if self.control == 'iasa' else self._assemble_obs()
        wp.synchronize()
        return np.clip(o.cpu().numpy(), -1, 1)

    def _assemble_obs(self):
        d = self._dev
        st = d["sense"]  # sensor space; dead lanes frozen at last sensed frame
        om, iabc, eps = st[:, 0], st[:, 1:4], st[:, 4]
        clk = d["clk"].clamp(max=d["dur"])
        ref = d["ref_tab"][clk, torch.arange(self.n, device=clk.device)]
        isd, isq = _dq_f32(iabc, eps)
        v = torch.stack([om / 25, ref / 25, (ref - om) / 25, isd / 4, isq / 4,
                         d["pcmd"] / 1.5, d["z"].to(torch.float32)], dim=1)
        return v.to(torch.float32)

    def step(self, actions):
        d = self._dev
        dev = d["plant"].device
        a = np.asarray(actions, dtype=float).reshape(self.n, -1)
        cmd = torch.as_tensor(1.5 * np.clip(a[:, 0], -1, 1),
                              dtype=torch.float32, device=dev)
        ar = d["ar_N"]
        with torch.no_grad(), torch.cuda.stream(self.tstream):
            if self.control == 'iasa':
                if a.shape[1] != 2:
                    raise ValueError('IASA control needs two actor outputs')
                cmdf, iasa_delta = self._iasa_command(a)
            else:
                d["cmd_f"] = self.alpha * cmd + (1.0 - self.alpha) * d["cmd_f"]
                cmdf = d["cmd_f"]
            clk0 = d["clk"].clone()
            ref0 = d["ref_tab"][clk0.clamp(max=d["dur"]), ar].to(torch.float64)
            om_pre = d["plant"][:, 0].clone()
            z0 = d["z"].clone()
            if self.control == 'iasa':
                delta = iasa_delta
                if self.shape == 'candidate':
                    dev = d["plant"].device
                    aP = torch.as_tensor(np.clip(a[:, 0], -1, 1), dtype=torch.float64, device=dev)
                    aI = torch.as_tensor(np.clip(a[:, 1], -1, 1), dtype=torch.float64, device=dev)
                    c_now = d["cmd_f"].double()
                    c_prev = d["pcmd"].double()
            else:
                delta = (cmd.double() - d["pcmd"].double()).abs() / 3.0
            d["rew"].zero_()
            d["cnt"].zero_()
            d["tprev"].copy_(self._torque_of(d["plant"]))
            for _ in range(self.hold):
                live = (~d["done"]) & (d["clk"] < d["dur"])
                d["live"].copy_(live.to(torch.int32))
                dist = d["dist_tab"][d["clk"].clamp(max=d["dur"]), ar]
                d["dist_b"].copy_(dist)
                i_f32 = d["sense"][:, 1:4].to(torch.float32)
                e_f32 = d["sense"][:, 4].to(torch.float32)
                o_f32 = d["sense"][:, 0].to(torch.float32)
                volt, branches, isd, isq = _inner_c(
                    i_f32, e_f32, cmdf, d["prev"], d["hist"], o_f32,
                    self.cur_lim, self.n, self.H, self.model.actor, self.ANL)
                d["prev"].copy_(torch.where(live.view(-1, 1), branches, d["prev"]))
                rolled = torch.roll(d["hist"], shifts=-1, dims=1)
                newdq = torch.stack([isd, isq], dim=1)
                rolled[:, -1, :] = torch.where(live.view(-1, 1), newdq, rolled[:, -1, :])
                d["hist"].copy_(torch.where(live.view(-1, 1, 1), rolled, d["hist"]))
                if self.sense.volt_delay:
                    # Transport delay: physics sees the previous command.
                    u_app = d["volt_prev"].clone()
                    d["volt_prev"].copy_(torch.where(live.view(-1, 1), volt.to(torch.float64),
                                                     d["volt_prev"]))
                    d["volt"].copy_(u_app)
                else:
                    d["volt"].copy_(volt.to(torch.float64))
                wp.launch(fs.fullstep_masked, dim=self.n,
                          inputs=[d["w_plant"], d["w_volt"], d["w_dist"], d["w_live"],
                                  wp.float64(self.tau_i), d["w_out"], d["w_params"]],
                          stream=self.wstream)
                # Stream-ordered: torch observes kernel output on the same stream.
                # Trip uses TRUE currents (protection); control/reward use sensed.
                om_n, i_n = d["out"][:, 0], d["out"][:, 1:4]
                ab = i_n.abs()
                tripped = ((ab[:, 0] > 4.0) | (ab[:, 1] > 4.0) | (ab[:, 2] > 4.0)) & live
                first = tripped & (~d["done"])
                d["snap"].copy_(torch.where(first.view(-1, 1), d["out"][:, :5], d["snap"]))
                d["fail"].copy_(torch.where(first, torch.ones_like(d["fail"]), d["fail"]))
                d["plant"].copy_(torch.where(live.view(-1, 1), d["out"][:, :5], d["plant"]))
                d["clk"] += live.to(torch.int64)
                d["cnt"] += live.to(torch.int64)
                self._sense_refresh(live)
                om_s, i_s, eps_s = self._sensed_post()
                isd_s, isq_s = _dq_f64(i_s, eps_s)
                tq_s = self._torque_of(d["sense"])
                if self.shape == 'candidate':
                    # Per-outer command costs are constant across inner samples;
                    # adding them per sample then taking the mean is exact.
                    d["rew"] += live.to(torch.float64) * (-_candidate_reward_c(
                        ref0, om_s, c_now, c_prev, aP, aI))
                else:
                    d["rew"] += live.to(torch.float64) * (-self._reward_terms(
                        ref0, om_s, isd_s, isq_s, delta, z0, tq_s, d["tprev"]))
                d["tprev"].copy_(tq_s)
                d["done"] |= tripped
            cnt = d["cnt"].clamp(min=1)
            rew = torch.where(d["cnt"] > 0, d["rew"] / cnt.to(torch.float64),
                              torch.zeros_like(d["rew"]))
            term = d["done"].clone()
            rew = torch.where(term, torch.full_like(rew, self.failure), rew)
            n_exec = d["cnt"].clone()
            if self.control == 'direct':
                d["z"] = torch.clamp(
                    z0 + n_exec.to(torch.float64) * self.tau_i
                    * torch.clamp((ref0 - om_pre) / 25, -1, 1) / self.mem_div, -1, 1)
                d["pcmd"].copy_(cmd)
            fin = (~term) & (d["clk"] >= d["dur"])
            trunc = fin
            newt = fin & (~term)
            d["snap"][newt] = d["plant"][newt]
            d["done"] |= trunc
            if self.control == 'iasa':
                obs_t = self._assemble_iasa_obs()
                self._record_iasa_history()
            else:
                obs_t = self._assemble_obs()
            fail = d["fail"].clone()
        wp.synchronize()
        obs = np.clip(obs_t.cpu().numpy(), -1, 1).astype(np.float32)
        rew_h = rew.cpu().numpy()
        term_h = term.cpu().numpy()
        trunc_h = trunc.cpu().numpy()
        delta_h = n_exec.cpu().numpy()
        fail_h = fail.cpu().numpy()
        self.phys += delta_h
        self.done = term_h | trunc_h
        infos = [{"failure": "phase_current_trip" if int(f) == 1 else None,
                  "physics_steps": int(p), "inner_steps": int(k),
                  "scenario": self.case_names[j]}
                 for j, (f, p, k) in enumerate(zip(fail_h, self.phys, delta_h))]
        return obs, rew_h, term_h, trunc_h, infos

    def _sense_refresh(self, live):
        """Post-physics sensed mirror. N0 is exactly identity (legacy)."""
        from warp_backend.sensing import sense_currents_torch
        import torch
        d, cfg = self._dev, self.sense
        dev = d["plant"].device
        if self.sense_gen is None:
            self.sense_gen = torch.Generator(device='cuda')
        n = self.n
        if cfg.i_noise > 0:
            na = torch.empty((n,), dtype=torch.float64, device=dev).normal_(
                0.0, cfg.i_noise, generator=self.sense_gen)
            nb = torch.empty((n,), dtype=torch.float64, device=dev).normal_(
                0.0, cfg.i_noise, generator=self.sense_gen)
        else:
            na = torch.zeros((n,), dtype=torch.float64, device=dev)
            nb = torch.zeros((n,), dtype=torch.float64, device=dev)
        sa, sb, sc = sense_currents_torch(cfg, torch, d["plant"][:, 1], d["plant"][:, 2],
                                          d["plant"][:, 3], na, nb, d["ifilt"])
        lv = live.view(-1, 1)
        d["ifilt"].copy_(torch.where(lv, torch.stack([sa, sb], dim=1), d["ifilt"]))
        new_hist = torch.roll(d["eps_hist"], shifts=-1, dims=1)
        new_hist[:, -1] = d["plant"][:, 4] + cfg.ang_off
        d["eps_hist"].copy_(torch.where(lv, new_hist, d["eps_hist"]))
        eps_s = d["eps_hist"][:, -1 - cfg.ang_age]
        d["sense"].copy_(torch.where(
            lv, torch.stack([d["plant"][:, 0], sa, sb, sc, eps_s], dim=1), d["sense"]))

    def _sensed_post(self):
        """Post-step sensed om/currents/eps for reward/torque (sensor space)."""
        d = self._dev
        return d["sense"][:, 0], d["sense"][:, 1:4], d["sense"][:, 4]

    def _torque_of(self, plant):
        om, i, eps = plant[:, 0], plant[:, 1:4], plant[:, 4]
        th = eps - torch.floor(eps / (2 * np.pi)) * (2 * np.pi)
        sh = 5 * np.pi / 6
        fa = _trap(_wrap(th + sh))
        fb = _trap(_wrap(th - 2 * np.pi / 3 + sh))
        fc = _trap(_wrap(th - 4 * np.pi / 3 + sh))
        ke = self._dev["ke"].to(plant.dtype)
        return ke * (fa * i[:, 0] + fb * i[:, 1] + fc * i[:, 2])

    def _reward_terms(self, ref, om, isd, isq, delta, z0, tq, tprev):
        mem = 0.0 if self.control == 'iasa' else self.mem_cost
        return _reward_c(ref, om, isd, isq, delta, z0, tq, tprev,
                         self.shape, self.qw, self.effort, mem)

    def _iasa_command(self, a):
        """Plan sect.2 action/state equations (device). Returns (cmdf, delta).
        Updates bias with anti-windup, filter state, and previous-applied."""
        from rl.outer.contract import (V3_DIRECT_GAIN_A, V3_INCREMENT_GAIN_A,
                                       V3_COMMAND_LIMIT_A, V3_ANTIWINDUP_GAIN)
        d = self._dev
        aP = torch.as_tensor(np.clip(a[:, 0], -1, 1), dtype=torch.float32, device=d["plant"].device)
        aI = torch.as_tensor(np.clip(a[:, 1], -1, 1), dtype=torch.float32, device=d["plant"].device)
        p = V3_DIRECT_GAIN_A * aP
        b_trial = d["bias"] + V3_INCREMENT_GAIN_A * aI
        v = p + b_trial
        lim = V3_COMMAND_LIMIT_A
        s = torch.clamp(v, -lim, lim)
        d["bias"] = torch.clamp(b_trial + V3_ANTIWINDUP_GAIN * (s - v), -lim, lim)
        c_prev = d["cmd_f"].clone()
        d["cmd_f"] = self.alpha * s + (1.0 - self.alpha) * d["cmd_f"]
        d["pcmd"].copy_(c_prev)
        delta = (d["cmd_f"].double() - c_prev.double()).abs() / 3.0
        return d["cmd_f"], delta

    def _assemble_iasa_obs(self):
        """19-dim v3 observation: 10 base + speed/error/cmd at 5/10/20 ms lags."""
        d = self._dev
        st = d["sense"]  # sensor space; dead lanes frozen at last sensed frame
        om, iabc, eps = st[:, 0], st[:, 1:4], st[:, 4]
        clk = d["clk"].clamp(max=d["dur"])
        ref = d["ref_tab"][clk, torch.arange(self.n, device=clk.device)]
        isd, isq = _dq_f32(iabc, eps)
        err = (ref - om) / 25
        base = torch.stack([om / 25, ref / 25, err, torch.tanh((ref - om) / 0.05),
                            isd / 4, isq / 4, d["bias"] / 1.5, d["pcmd"] / 1.5,
                            torch.sin(eps), torch.cos(eps)], dim=1)
        lags = torch.stack([d["h_spd"][:, -5] / 25, d["h_err"][:, -5] / 25, d["h_cmd"][:, -5] / 1.5,
                            d["h_spd"][:, -10] / 25, d["h_err"][:, -10] / 25, d["h_cmd"][:, -10] / 1.5,
                            d["h_spd"][:, -20] / 25, d["h_err"][:, -20] / 25,
                            d["h_cmd"][:, -20] / 1.5], dim=1)
        v = torch.cat([base, lags], dim=1)
        if not torch.isfinite(v).all():
            raise ValueError('Nonfinite IASA observation')
        return torch.clamp(v, -1, 1).to(torch.float32)

    def _record_iasa_history(self):
        """Record current frame; called after obs assembly each outer step."""
        d = self._dev
        om = d["sense"][:, 0]  # measured speed history, not ground truth
        clk = d["clk"].clamp(max=d["dur"])
        ref = d["ref_tab"][clk, torch.arange(self.n, device=clk.device)]
        d["h_spd"].copy_(torch.roll(d["h_spd"], shifts=-1, dims=1))
        d["h_err"].copy_(torch.roll(d["h_err"], shifts=-1, dims=1))
        d["h_cmd"].copy_(torch.roll(d["h_cmd"], shifts=-1, dims=1))
        d["h_spd"][:, -1] = om.float()
        d["h_err"][:, -1] = (ref - om).float()
        d["h_cmd"][:, -1] = d["cmd_f"]

import os as _os

_COMPILE = _os.environ.get("VROOM_NO_COMPILE", "0") != "1"


def _maybe_compile(fn):
    # reduce-overhead replays compiled regions as CUDA graphs: dynamo guard
    # evaluation (~110us/call) otherwise dominates these tiny regions.
    # Static shapes per backend instance; no data-dependent flow inside.
    return torch.compile(fn, mode="reduce-overhead") if _COMPILE else fn


def _inner_block(iabc, eps, cmdf, prev, hist, om, cur_lim, n, H, actor, anl):
    """Fused per-inner-step controller: dq, encode, frozen actor, project.

    One compiled region instead of separate encode/actor/project calls plus
    eager glue, cutting dynamo per-call overhead and torch dispatch count.
    The actor module is closed over (frozen inner policy); guards specialize
    on its static state.
    """
    e = eps.to(torch.float32)
    a = (2.0 / 3.0) * (iabc[:, 0] - 0.5 * iabc[:, 1] - 0.5 * iabc[:, 2])
    b = (2.0 / 3.0) * (0.8660254037844386 * iabc[:, 1] - 0.8660254037844386 * iabc[:, 2])
    ce, se = torch.cos(e), torch.sin(e)
    isd, isq = ce * a + se * b, -se * a + ce * b
    s = 4.0
    i = torch.stack([isd, isq], dim=1)
    ref = torch.stack([torch.zeros_like(cmdf),
                       cmdf * torch.clamp(cur_lim / torch.clamp(cmdf.abs(), min=1e-30),
                                          max=1.0)], dim=1)
    ep = eps.to(torch.float32)
    vals = torch.cat([i / s, ref / s, (ref - i) / (2 * s), prev,
                      hist.reshape(n, -1) / s, (om / 20.0).unsqueeze(1),
                      torch.sin(ep).unsqueeze(1), torch.cos(ep).unsqueeze(1),
                      torch.zeros((n, 2), dtype=torch.float32, device=om.device)], dim=1)
    obs_in = torch.clamp(vals, -1, 1)
    raw = actor(obs_in)
    r = torch.clamp(raw, -1, 1)
    branches = torch.cat([r, torch.zeros((n, 2), dtype=r.dtype, device=r.device)], dim=1)
    requested = r * anl
    nrm = torch.sqrt(requested[:, 0] ** 2 + requested[:, 1] ** 2).unsqueeze(1)
    scale = torch.minimum(torch.ones_like(nrm), anl / torch.clamp(nrm, min=1e-30))
    return requested * scale, branches, isd, isq


def _candidate_reward_vals(ref, om, c_now, c_prev, a_p, a_i):
    """Plan sect.3 candidate cost per inner sample (v4 contract).

    Tracking is per executed inner sample; the outer command costs are
    constant across the hold, so callers may add them per sample before
    taking the mean over executed steps. Mirrors rl.outer.reward reference.
    """
    e = torch.clamp(ref - om, -25.0, 25.0)
    track = 4.0 * (e / 25.0) ** 2 + 0.1 * torch.log1p((e / 0.05) ** 2)
    cmd = (0.02 * ((c_now - c_prev) / 3.0) ** 2
           + 0.01 * a_i ** 2 + 0.001 * a_p ** 2)
    return track + cmd


def _reward_vals(ref, om, isd, isq, delta, z0, tq, tprev, shape, qw, effort, mem_cost):
    en = torch.clamp((ref - om) / 25, -1, 1)
    if shape == "qeff":
        spd = qw * en ** 2
    else:
        spd = 4.0 * en ** 2
        if shape == "l1":
            spd = spd + 4.0 * (en.abs() - en ** 2)
    cur = 0.5 * torch.clamp(torch.hypot(isd, isq) / 4, 0, 1) ** 2
    dlt = effort * 0.1 * torch.clamp(delta, 0, 1) ** 2
    mem = mem_cost * z0 ** 2
    tqd = effort * 0.1 * torch.clamp((tq - tprev).abs() / 0.1, 0, 1) ** 2
    return spd + cur + dlt + mem + tqd


_reward_c = _maybe_compile(_reward_vals)
_candidate_reward_c = _maybe_compile(_candidate_reward_vals)
_inner_c = _maybe_compile(_inner_block)


def _wrap(t):
    return t - torch.floor(t / (2 * np.pi)) * (2 * np.pi)


def _trap(t):
    lo1, pi, hi5, tpi = 2 * np.pi / 3, np.pi, 5 * np.pi / 3, 2 * np.pi
    ramp_down = 1.0 - 2.0 * (t - lo1) / (pi - lo1)
    ramp_up = -1.0 + 2.0 * (t - hi5) / (tpi - hi5)
    mid = torch.where(t < pi, ramp_down, torch.tensor(-1.0, device=t.device))
    tail = torch.where(t < hi5, mid, ramp_up)
    return torch.where(t < lo1, torch.tensor(1.0, device=t.device), tail)


def _dq_f32(iabc, eps):
    e = eps.to(torch.float32)
    a = (2.0 / 3.0) * (iabc[:, 0] - 0.5 * iabc[:, 1] - 0.5 * iabc[:, 2])
    b = (2.0 / 3.0) * (SQRT3_2 * iabc[:, 1] - SQRT3_2 * iabc[:, 2])
    ce, se = torch.cos(e), torch.sin(e)
    return ce * a + se * b, -se * a + ce * b


def _dq_f64(iabc, eps):
    a = (2.0 / 3.0) * (iabc[:, 0] - 0.5 * iabc[:, 1] - 0.5 * iabc[:, 2])
    b = (2.0 / 3.0) * (SQRT3_2 * iabc[:, 1] - SQRT3_2 * iabc[:, 2])
    ce, se = torch.cos(eps), torch.sin(eps)
    return ce * a + se * b, -se * a + ce * b

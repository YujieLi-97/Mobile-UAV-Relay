#!/usr/bin/env python3
"""Current UAV relay experiment in one runnable file.

Commands: calibrate, calibrate_extra, graph_calibrate, select, confirm,
audit, stress, summarize, assets, seed-audit, verify, and all.
The module sections below are kept verbatim from the validated experiment.
"""

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parent
MODULE_SOURCE = {
    "sim": r'''

import numpy as np
from scipy.ndimage import gaussian_filter


class Cfg:

    L = 2000.0
    H_UAV = 100.0
    GRID = 101


    DT = 1.0
    T_STEPS = 400


    V_MAX = 20.0
    A_MAX = 4.0


    V_G = 5.0
    G_HEADING_STD = 0.06


    BS = np.array([1000.0, 1000.0, 0.0])


    P_TX_DBM = 30.0
    BETA0_DB = -40.0
    ALPHA_P = 2.2
    BW = 1e6
    NOISE_DBM = -100.0
    # 第四轮主实验会覆盖这个旧实验默认门限。
    R_MIN = 4.5e6


    PJ_DBM = 24.0
    ALPHA_J = 3.0
    SHADOW_STD = 6.0
    SHADOW_CORR = 150.0
    N_BLOCK = 3
    BLOCK_DB = 12.0


    SIGMA_Y = 1.0


    USE_FADING = False
    RICIAN_K_DB = 10.0


    P_BLADE = 79.86
    P_INDUCED = 88.63
    U_TIP = 120.0
    V_0 = 4.03
    D_0 = 0.6
    RHO_AIR = 1.225
    S_ROTOR = 0.05
    A_ROTOR = 0.503


class TrueField:


    def __init__(self, rng: np.random.Generator, cfg=Cfg):
        self.cfg = cfg
        self.rng = rng


        # 干扰机坐标是每个场景随机流的前两个抽样值。
        m = 0.15 * cfg.L
        self.p_J = rng.uniform(m, cfg.L - m, size=2)


        ax = np.linspace(0.0, cfg.L, cfg.GRID)
        self.ax = ax
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        self.cell = ax[1] - ax[0]


        d = np.hypot(X - self.p_J[0], Y - self.p_J[1])
        d = np.maximum(d, 1.0)
        path = cfg.PJ_DBM + cfg.BETA0_DB - 10.0 * cfg.ALPHA_J * np.log10(d)


        sigma_pix = cfg.SHADOW_CORR / self.cell / 2.0
        white = rng.standard_normal((cfg.GRID, cfg.GRID))
        sh = gaussian_filter(white, sigma=sigma_pix, mode="wrap")
        sh = sh / (sh.std() + 1e-12) * cfg.SHADOW_STD


        block = np.zeros_like(path)
        for _ in range(cfg.N_BLOCK):
            cx, cy = rng.uniform(0.2 * cfg.L, 0.8 * cfg.L, size=2)
            w, h = rng.uniform(0.08 * cfg.L, 0.20 * cfg.L, size=2)
            mask = (np.abs(X - cx) < w / 2) & (np.abs(Y - cy) < h / 2)
            block[mask] -= cfg.BLOCK_DB

        self.residual = sh + block
        self.grid = path + self.residual


    def __call__(self, pts):

        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        u = np.clip(pts[:, 0] / self.cell, 0, self.cfg.GRID - 1.001)
        v = np.clip(pts[:, 1] / self.cell, 0, self.cfg.GRID - 1.001)
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        du, dv = u - i0, v - j0
        # 阴影残差在水平面插值，距离衰减再使用接收点的实际高度。
        g = self.residual
        residual = (g[i0, j0] * (1 - du) * (1 - dv)
                + g[i0 + 1, j0] * du * (1 - dv)
                + g[i0, j0 + 1] * (1 - du) * dv
                + g[i0 + 1, j0 + 1] * du * dv)
        z = pts[:, 2] if pts.shape[1] == 3 else np.zeros(len(pts))
        d = np.sqrt((pts[:, 0] - self.p_J[0]) ** 2
                    + (pts[:, 1] - self.p_J[1]) ** 2 + z ** 2)
        path = self.cfg.PJ_DBM + self.cfg.BETA0_DB - 10.0 * self.cfg.ALPHA_J * np.log10(np.maximum(d, 1.0))
        return path + residual


def make_ground_track(rng, cfg=Cfg):

    edge = rng.integers(0, 4)
    t0 = rng.uniform(0.15, 0.85) * cfg.L
    t1 = rng.uniform(0.15, 0.85) * cfg.L
    if edge == 0:      start, goal = np.array([0.0, t0]),    np.array([cfg.L, t1])
    elif edge == 1:    start, goal = np.array([cfg.L, t0]),  np.array([0.0, t1])
    elif edge == 2:    start, goal = np.array([t0, 0.0]),    np.array([t1, cfg.L])
    else:              start, goal = np.array([t0, cfg.L]),  np.array([t1, 0.0])

    theta = np.arctan2(*(goal - start)[::-1])
    pos = start.copy()
    traj = [pos.copy()]
    for _ in range(cfg.T_STEPS):
        theta += rng.normal(0.0, cfg.G_HEADING_STD)
        pos = pos + cfg.V_G * cfg.DT * np.array([np.cos(theta), np.sin(theta)])
        pos = np.clip(pos, 0.0, cfg.L)
        traj.append(pos.copy())
    return np.array(traj)


def dbm2mw(x):
    return 10.0 ** (np.asarray(x) / 10.0)


def rx_power_dbm(d3, cfg=Cfg):

    d3 = np.maximum(d3, 1.0)
    return cfg.P_TX_DBM + cfg.BETA0_DB - 10.0 * cfg.ALPHA_P * np.log10(d3)


def link_rate(pr_dbm, interf_dbm, cfg=Cfg):

    s = dbm2mw(pr_dbm)
    i = dbm2mw(interf_dbm) + dbm2mw(cfg.NOISE_DBM)
    return cfg.BW * np.log2(1.0 + s / i)


def e2e_rate(q_xy, g_xy, phi_q, phi_g, cfg=Cfg):

    d_bu = np.sqrt(np.sum((q_xy - cfg.BS[:2]) ** 2, axis=-1) + cfg.H_UAV ** 2)
    d_ug = np.sqrt(np.sum((q_xy - g_xy) ** 2, axis=-1) + cfg.H_UAV ** 2)
    r1 = link_rate(rx_power_dbm(d_bu, cfg), phi_q, cfg)
    r2 = link_rate(rx_power_dbm(d_ug, cfg), phi_g, cfg)
    return 0.5 * np.minimum(r1, r2)


def i_max_dbm(pr_dbm, cfg=Cfg):

    denom = 2.0 ** (2.0 * cfg.R_MIN / cfg.BW) - 1.0
    i_mw = dbm2mw(pr_dbm) / denom - dbm2mw(cfg.NOISE_DBM)
    out = np.full(np.shape(i_mw), -np.inf)
    ok = i_mw > 0
    out[ok] = 10.0 * np.log10(i_mw[ok])
    return out


def prop_power(v, cfg=Cfg):

    v = np.atleast_1d(np.asarray(v, dtype=float))
    blade = cfg.P_BLADE * (1.0 + 3.0 * v ** 2 / cfg.U_TIP ** 2)
    induced = cfg.P_INDUCED * np.sqrt(
        np.maximum(np.sqrt(1.0 + v ** 4 / (4.0 * cfg.V_0 ** 4))
                   - v ** 2 / (2.0 * cfg.V_0 ** 2), 0.0))
    parasite = 0.5 * cfg.D_0 * cfg.RHO_AIR * cfg.S_ROTOR * cfg.A_ROTOR * v ** 3
    return blade + induced + parasite


def hover_power(cfg=Cfg):

    return cfg.P_BLADE + cfg.P_INDUCED


def rician_gain_db(rng, size, cfg=Cfg):

    k = 10.0 ** (cfg.RICIAN_K_DB / 10.0)
    los = np.sqrt(k / (k + 1.0))
    nlos = np.sqrt(1.0 / (2.0 * (k + 1.0)))
    h = los + nlos * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
    return 10.0 * np.log10(np.abs(h) ** 2)
''',
    "planner": r'''

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_triangular, cho_solve
from scipy.spatial.distance import cdist

from sim import (Cfg, rx_power_dbm, i_max_dbm, prop_power,
                 hover_power, dbm2mw)


class GaussianField:
    def __init__(self, ell=250.0, sigma_f=12.0, sigma_y=1.0,
                 mean_dbm=-105.0, max_obs=150):
        self.ell, self.sf, self.sy = ell, sigma_f, sigma_y
        self.m0 = mean_dbm
        self.max_obs = max_obs
        self.X = np.zeros((0, 3))
        self.y = np.zeros(0)
        self._dirty = True
        self._L = None
        self._alpha = None


    def _k(self, A, B):
        d = cdist(A, B)
        r = np.sqrt(5.0) * d / self.ell
        return self.sf ** 2 * (1.0 + r + r ** 2 / 3.0) * np.exp(-r)

    def add(self, x, y):

        self.X = np.vstack([self.X, np.atleast_2d(x)])
        self.y = np.append(self.y, y)
        # 只保留最近的读数，限制 GP 矩阵规模。
        if len(self.y) > self.max_obs:
            idx = np.arange(len(self.y) - self.max_obs, len(self.y))
            self.X, self.y = self.X[idx], self.y[idx]
        self._dirty = True

    def _fit(self):
        n = len(self.y)
        if n == 0:
            return
        K = self._k(self.X, self.X) + (self.sy ** 2) * np.eye(n)
        self._L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        self._alpha = cho_solve((self._L, True), self.y - self.m0, check_finite=False)
        self._dirty = False

    def predict(self, Q):

        Q = np.atleast_2d(Q)
        if len(self.y) == 0:
            return (np.full(len(Q), self.m0),
                    np.full(len(Q), self.sf))
        if self._dirty:
            self._fit()
        Ks = self._k(Q, self.X)
        mu = self.m0 + Ks @ self._alpha
        v = solve_triangular(self._L, Ks.T, lower=True, check_finite=False)
        var = self.sf ** 2 - np.sum(v ** 2, axis=0)
        return mu, np.sqrt(np.maximum(var, 1e-9))

    def update_hyper(self, ell_grid=None, sf_grid=None, ell_min=50.0):

        n = len(self.y)
        if n < 5:
            return
        ell_grid = ell_grid if ell_grid is not None else \
            np.array([100., 175., 250., 350., 500.])
        sf_grid = sf_grid if sf_grid is not None else np.array([6., 9., 12., 16.])
        ell_grid = ell_grid[ell_grid >= ell_min]
        best, arg = -np.inf, (self.ell, self.sf)
        y0 = self.y - self.m0
        for ell in ell_grid:
            for sf in sf_grid:
                old = (self.ell, self.sf)
                self.ell, self.sf = ell, sf
                K = self._k(self.X, self.X) + (self.sy ** 2) * np.eye(n)
                try:
                    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                except np.linalg.LinAlgError:
                    self.ell, self.sf = old
                    continue
                a = cho_solve((L, True), y0, check_finite=False)

                lml = -0.5 * y0 @ a - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                self.ell, self.sf = old
                if lml > best:
                    best, arg = lml, (ell, sf)
        self.ell, self.sf = arg
        self._dirty = True

    def argmax_grid(self, n=41, L=None):

        L = Cfg.L if L is None else L
        ax = np.linspace(0, L, n)
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        mu, _ = self.predict(P)
        return P[int(np.argmax(mu)), :2]


def outage_prob(q_xy, g_xy, mu_q, sd_q, mu_g, sd_g, cfg=Cfg,
                use_chance=True, sd_floor=1.0):

    q_xy = np.atleast_2d(q_xy)
    g_xy = np.atleast_2d(g_xy)
    d_bu = np.sqrt(((q_xy - cfg.BS[:2]) ** 2).sum(-1) + cfg.H_UAV ** 2)
    d_ug = np.sqrt(((q_xy - g_xy) ** 2).sum(-1) + cfg.H_UAV ** 2)

    need_db = 10.0 * np.log10(2.0 ** (2.0 * cfg.R_MIN / cfg.BW) - 1.0)
    n_mw = dbm2mw(cfg.NOISE_DBM)

    p_fail = []
    for d, mu, sd in ((d_bu, mu_q, sd_q), (d_ug, mu_g, sd_g)):
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        sd = np.atleast_1d(np.asarray(sd, dtype=float))
        if not use_chance:
            sd = np.full_like(mu, sd_floor)

        pr_dbm = rx_power_dbm(d, cfg)
        i_mw = dbm2mw(mu)
        tot_mw = i_mw + n_mw
        sinr_db = pr_dbm - 10.0 * np.log10(tot_mw)
        margin = sinr_db - need_db

        frac = i_mw / tot_mw
        sigma_eff = np.sqrt((sd * frac) ** 2 + sd_floor ** 2)
        p_fail.append(np.clip(1.0 - norm.cdf(margin / sigma_eff), 0.0, 1.0))

    return 1.0 - (1.0 - p_fail[0]) * (1.0 - p_fail[1])


class MPPIPlanner:


    def __init__(self, lam=0.0, horizon=10, n_samples=150, temperature=1.0,
                 sigma_v=8.0, oracle_fn=None, cfg=Cfg, lookahead=60,
                 seed=0, w_energy=0.0, use_chance=True, a_max=None,
                 w_term=None, n_term=10):
        self.lam, self.H, self.K = lam, horizon, n_samples
        self.w_energy = w_energy
        self.use_chance = use_chance
        self.a_max = a_max


        self.w_term = horizon if w_term is None else w_term
        self.n_term = n_term
        self.lookahead = lookahead
        self.temp, self.sigma_v = temperature, sigma_v
        self.oracle_fn = oracle_fn
        self.cfg = cfg
        self.last_velocity = np.zeros(2)
        self.u_prev = np.zeros((horizon, 2))
        self.rng = np.random.default_rng(seed)

    def _predict(self, gp, P):
        if self.oracle_fn is not None:


            return self.oracle_fn(P), np.full(len(P), gp.sy)
        return gp.predict(P)

    def step(self, q, gp, g_now, g_vel):

        cfg, H, K = self.cfg, self.H, self.K
        rng = self.rng


        g_pred = g_now + np.outer(np.arange(1, H + 1) * cfg.DT, g_vel)


        noise = rng.normal(0.0, self.sigma_v, size=(K, H, 2))
        U = self.u_prev[None] + noise

        # 候选速度逐步投影到加速度允许的圆盘内。
        if self.a_max is not None:
            lim = self.a_max * cfg.DT
            prev = np.tile(self.last_velocity, (K, 1))
            for h in range(H):
                dv = U[:, h] - prev
                dn = np.linalg.norm(dv, axis=-1, keepdims=True)
                dv = np.where(dn > lim, dv / np.maximum(dn, 1e-9) * lim, dv)
                U[:, h] = prev + dv
                prev = U[:, h]

        nrm = np.linalg.norm(U, axis=-1, keepdims=True)
        U = np.where(nrm > cfg.V_MAX, U / np.maximum(nrm, 1e-9) * cfg.V_MAX, U)


        Q = q[None, None, :] + np.cumsum(U * cfg.DT, axis=1)
        Q = np.clip(Q, 0.0, cfg.L)


        flatQ = Q.reshape(-1, 2)
        mu_q, sd_q = self._predict(gp, np.column_stack([flatQ, np.full(len(flatQ), cfg.H_UAV)]))
        mu_g, sd_g = self._predict(gp, np.column_stack([g_pred, np.zeros(len(g_pred))]))
        mu_g = np.tile(mu_g, K)
        sd_g = np.tile(sd_g, K)
        G = np.tile(g_pred, (K, 1))


        p_out = outage_prob(flatQ, G, mu_q, sd_q, mu_g, sd_g, cfg,
                            use_chance=self.use_chance)


        g_far = g_now + np.outer(
            np.arange(1, self.lookahead + 1) * cfg.DT, g_vel)
        g_far = np.clip(g_far, 0.0, cfg.L)
        d_far = cdist(flatQ, g_far)
        relevance = np.exp(-(d_far.min(axis=1) / (1.5 * gp.ell)) ** 2)
        info_gain = (sd_q / gp.sf) * relevance


        if self.w_energy > 0.0:
            speeds = np.linalg.norm(U, axis=-1).reshape(-1)
            c_energy = prop_power(speeds, cfg) / hover_power(cfg)
        else:
            c_energy = 0.0

        run_cost = (p_out - self.lam * info_gain
                    + self.w_energy * c_energy).reshape(K, H).sum(axis=1)


        if self.w_term > 0.0:
            step = max(1, len(g_far) // self.n_term)
            gf = g_far[::step][:self.n_term]
            Fs = len(gf)
            qT = Q[:, -1, :]
            mu_T, sd_T = self._predict(gp, np.column_stack([qT, np.full(len(qT), cfg.H_UAV)]))
            mu_f, sd_f = self._predict(gp, np.column_stack([gf, np.zeros(len(gf))]))
            QT = np.repeat(qT, Fs, axis=0)
            GF = np.tile(gf, (K, 1))
            p_term = outage_prob(QT, GF,
                                 np.repeat(mu_T, Fs), np.repeat(sd_T, Fs),
                                 np.tile(mu_f, K), np.tile(sd_f, K),
                                 cfg, use_chance=self.use_chance)
            cost = run_cost + self.w_term * p_term.reshape(K, Fs).mean(axis=1)
        else:
            cost = run_cost


        # 平移代价后再计算权重，避免指数计算溢出。
        w = np.exp(-(cost - cost.min()) / self.temp)
        w /= w.sum()
        u_new = np.einsum("k,khd->hd", w, U)


        self.u_prev = np.vstack([u_new[1:], u_new[-1:]])
        return u_new[0]


def oblivious_velocity(q, g_now, cfg=Cfg):

    target = 0.5 * (cfg.BS[:2] + g_now)
    d = target - q
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.zeros(2)
    return d / n * min(cfg.V_MAX, n / cfg.DT)


def explore_velocity(q, gp, cfg=Cfg, n_dir=16):

    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    cand = q + cfg.V_MAX * cfg.DT * np.column_stack([np.cos(ang), np.sin(ang)])
    cand = np.clip(cand, 0.0, cfg.L)
    _, sd = gp.predict(np.column_stack([cand, np.full(len(cand), cfg.H_UAV)]))
    best = cand[int(np.argmax(sd))]
    d = best - q
    n = np.linalg.norm(d)
    return np.zeros(2) if n < 1e-6 else d / n * cfg.V_MAX
''',
    "baselines": r'''

import numpy as np

from sim import Cfg, e2e_rate
from planner import oblivious_velocity


class ReactiveBaseline:


    def __init__(self, cfg=Cfg, memory=40):
        self.cfg = cfg
        self.good = []
        self.memory = memory

    def act(self, q, g_now, last_ok):

        cfg = self.cfg
        if last_ok:
            self.good.append(q.copy())
            if len(self.good) > self.memory:
                self.good.pop(0)
            return oblivious_velocity(q, g_now, cfg)

        if not self.good:
            return oblivious_velocity(q, g_now, cfg)


        target = self.good[-1]
        d = target - q
        n = np.linalg.norm(d)
        if n < 1.0:
            if len(self.good) > 1:
                self.good.pop()
            return oblivious_velocity(q, g_now, cfg)
        return d / n * min(cfg.V_MAX, n / cfg.DT)


def best_static_hover(field, gtraj, cfg=Cfg, grid=41):

    ax = np.linspace(0.0, cfg.L, grid)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    cand = np.column_stack([X.ravel(), Y.ravel()])
    phi_c = field(np.column_stack([cand, np.full(len(cand), cfg.H_UAV)]))

    gpos = gtraj[:cfg.T_STEPS]
    phi_g = field(gpos)

    best_rate, best_pos = -1.0, cand[0]
    for i in range(len(cand)):
        q = np.tile(cand[i], (len(gpos), 1))
        r = e2e_rate(q, gpos, np.full(len(gpos), phi_c[i]), phi_g, cfg)
        ok = float(np.mean(r >= cfg.R_MIN))
        if ok > best_rate:
            best_rate, best_pos = ok, cand[i]
    return best_pos, 1.0 - best_rate


def infeasible_fraction(field, gtraj, cfg=Cfg, grid=31):

    ax = np.linspace(0.0, cfg.L, grid)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    cand = np.column_stack([X.ravel(), Y.ravel()])
    phi_c = field(np.column_stack([cand, np.full(len(cand), cfg.H_UAV)]))
    gpos = gtraj[:cfg.T_STEPS]
    phi_g = field(gpos)

    bad = 0
    for t in range(len(gpos)):
        q = np.tile(cand[None, 0], (len(cand), 1)) * 0 + cand
        g = np.tile(gpos[t], (len(cand), 1))
        r = e2e_rate(q, g, phi_c, np.full(len(cand), phi_g[t]), cfg)
        if not np.any(r >= cfg.R_MIN):
            bad += 1
    return bad / len(gpos)
''',
    "experiment": r'''

import time

import numpy as np
from scipy import stats as sps
from functools import lru_cache

from sim import (Cfg, TrueField, make_ground_track, e2e_rate,
                 prop_power, hover_power, rician_gain_db, dbm2mw)
from planner import (GaussianField, MPPIPlanner,
                     oblivious_velocity, explore_velocity)
from baselines import (ReactiveBaseline, best_static_hover,
                       infeasible_fraction)


METHODS = ("ours", "oracle", "oblivious", "twophase", "reactive", "static")


def run_episode(seed, method, lam=0.0, te_steps=60, cfg=Cfg, record=False,
                use_feedback=True, w_energy=0.0, use_chance=True,
                ell_gp=250.0, online_hyper=False, max_obs=150,
                lookahead=60, a_max=Cfg.A_MAX, use_fading=None, w_term=None, grid_reference=61):

    assert method in METHODS, f"未知方法 {method}"
    rng = np.random.default_rng(seed)
    field = TrueField(rng, cfg)
    gtraj = make_ground_track(rng, cfg)
    fading = cfg.USE_FADING if use_fading is None else use_fading

    # 按时隙固定测量噪声，便于不同路径做配对比较。
    obs_rng = np.random.default_rng(np.random.SeedSequence([seed, 173]))
    obs_noise = obs_rng.normal(0, cfg.SIGMA_Y, (cfg.T_STEPS, 2))
    fade_rng = np.random.default_rng(np.random.SeedSequence([seed, 271]))
    fades = rician_gain_db(fade_rng, (cfg.T_STEPS, 2), cfg)


    floor = infeasible_fraction(field, gtraj, cfg, grid=grid_reference)

    if method == "static":
        pos, outage = best_static_hover(field, gtraj, cfg, grid=grid_reference)
        qs = np.tile(pos, (cfg.T_STEPS, 1))
        phi_q = np.full(cfg.T_STEPS, float(field(np.r_[pos, cfg.H_UAV])[0]))
        phi_g = field(gtraj[:cfg.T_STEPS])
        rates = e2e_rate(qs, gtraj[:cfg.T_STEPS], phi_q, phi_g, cfg)
        if fading:
            rates = np.array([_rate_with_fading(pos, gtraj[k], phi_q[k], phi_g[k], fades[k], cfg) for k in range(cfg.T_STEPS)])
        return _finalize(rates, np.zeros(cfg.T_STEPS), [], [], cfg,
                         energy_j=hover_power(cfg) * cfg.DT * cfg.T_STEPS,
                         floor=floor, record=record, qs=qs, gtraj=gtraj,
                         field=field, gp=None)

    q = 0.5 * (cfg.BS[:2] + gtraj[0])
    gp = GaussianField(ell=ell_gp, sigma_y=cfg.SIGMA_Y, max_obs=max_obs)

    oracle_fn = (lambda P: field(P)) if method == "oracle" else None
    planner = MPPIPlanner(lam=(lam if method == "ours" else 0.0),
                          oracle_fn=oracle_fn, cfg=cfg, seed=seed,
                          w_energy=w_energy, use_chance=use_chance,
                          lookahead=lookahead, a_max=a_max, w_term=w_term)
    reactive = ReactiveBaseline(cfg) if method == "reactive" else None

    rates, solve_ms, speeds = [], [], []
    sd_at_g, loc_err, ahead_rmse = [], [], []
    qs = []
    last_ok = None
    v_prev = np.zeros(2)

    for k in range(cfg.T_STEPS):
        g_now = gtraj[k]
        g_vel = (g_now - gtraj[k - 1]) / cfg.DT if k > 0 else np.zeros(2)


        t0 = time.perf_counter()
        # 先选本时隙动作，执行后才产生新的无线读数。
        if method == "oblivious":
            v = oblivious_velocity(q, g_now, cfg)
        elif method == "reactive":
            v = reactive.act(q, g_now, last_ok)
        elif method == "twophase" and k < te_steps:
            v = explore_velocity(q, gp, cfg)
        else:
            v = planner.step(q, gp, g_now, g_vel)
        solve_ms.append((time.perf_counter() - t0) * 1e3)


        if a_max is not None:
            dv = v - v_prev
            dn = np.linalg.norm(dv)
            if dn > a_max * cfg.DT:
                v = v_prev + dv * (a_max * cfg.DT / dn)
        vn = np.linalg.norm(v)
        if vn > cfg.V_MAX:
            v = v * (cfg.V_MAX / vn)
        q_next = np.clip(q + v * cfg.DT, 0.0, cfg.L)
        v_prev = (q_next - q) / cfg.DT
        planner.last_velocity = v_prev.copy()
        q = q_next
        qs.append(q.copy())
        speeds.append(float(np.linalg.norm(v_prev)))


        phi_q = float(field(np.r_[q, cfg.H_UAV])[0])
        phi_g = float(field(g_now)[0])
        if fading:
            fade = fades[k]
            r = _rate_with_fading(q, g_now, phi_q, phi_g, fade, cfg)
        else:
            r = float(e2e_rate(q, g_now, phi_q, phi_g, cfg))
        rates.append(r)
        last_ok = r >= cfg.R_MIN


        if method != "oracle":
            update_allowed = not (method == "twophase" and k >= te_steps)
            if update_allowed:
                gp.add(np.r_[q, cfg.H_UAV], phi_q + obs_noise[k, 0])
                # 用户读数只在两跳服务成功时回传。
                if use_feedback and last_ok:
                    gp.add(np.r_[g_now, 0.0], phi_g + obs_noise[k, 1])
                if online_hyper and k % 20 == 19:
                    gp.update_hyper()


            _, sd = gp.predict(np.r_[g_now, 0.0][None, :])
            sd_at_g.append(float(sd[0]))

            if k % 10 == 0:
                loc_err.append(float(np.linalg.norm(
                    gp.argmax_grid() - field.p_J)))
                W = 30
                fut = gtraj[k + 1: k + 1 + W]
                if len(fut):
                    mu, _ = gp.predict(np.column_stack([fut, np.zeros(len(fut))]))
                    ahead_rmse.append(float(np.sqrt(
                        np.mean((mu - field(fut)) ** 2))))
        else:
            sd_at_g.append(0.0)

    energy_j = float(np.sum(prop_power(np.array(speeds), cfg)) * cfg.DT)
    return _finalize(np.array(rates), np.array(sd_at_g), loc_err, ahead_rmse,
                     cfg, energy_j=energy_j, floor=floor, solve_ms=solve_ms,
                     record=record, qs=np.array(qs), gtraj=gtraj,
                     field=field, gp=gp)


def _rate_with_fading(q, g, phi_q, phi_g, fade_db, cfg):

    from sim import rx_power_dbm, link_rate
    d_bu = np.sqrt(np.sum((q - cfg.BS[:2]) ** 2) + cfg.H_UAV ** 2)
    d_ug = np.sqrt(np.sum((q - g) ** 2) + cfg.H_UAV ** 2)
    r1 = link_rate(rx_power_dbm(d_bu, cfg) + fade_db[0], phi_q, cfg)
    r2 = link_rate(rx_power_dbm(d_ug, cfg) + fade_db[1], phi_g, cfg)
    return float(0.5 * min(r1, r2))


def _finalize(rates, sd_at_g, loc_err, ahead_rmse, cfg, energy_j, floor=0.0,
              solve_ms=None, record=False, qs=None, gtraj=None,
              field=None, gp=None):
    outages = rates < cfg.R_MIN
    longest, cur, runs = 0, 0, []
    for o in outages:
        if o:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
        longest = max(longest, cur)
    if cur:
        runs.append(cur)

    out = dict(
        outage_rate=float(outages.mean()),

        infeasible_rate=float(floor),
        grid_reference_rate=float(floor),


        excess_outage=float(outages.mean() - floor),
        longest_outage_s=float(longest * cfg.DT),
        rate_p05=float(np.percentile(rates, 5)),
        ahead_rmse=float(np.mean(ahead_rmse)) if ahead_rmse else float("nan"),
        final_loc_err=float(loc_err[-1]) if loc_err else float("nan"),
        energy_kj=energy_j / 1e3,
        solve_ms_med=float(np.median(solve_ms)) if solve_ms is not None and len(solve_ms) else 0.0,
    )
    if record:
        out["_rates"] = rates
        out["_outages"] = outages
        out["_runs_s"] = np.array(runs, dtype=float) * cfg.DT
        out["_sd_at_g"] = sd_at_g
        out["_traj"] = qs
        out["_gtraj"] = gtraj
        out["_field"] = field
        out["_gp"] = gp
        out["_ahead_rmse"] = np.array(ahead_rmse)
        out["_solve_ms"] = np.asarray(solve_ms if solve_ms is not None else [])
    return out


def run_many(method, seeds, **kw):

    rows = [run_episode(s, method, **kw) for s in seeds]
    out = {}
    for k in rows[0]:
        if k.startswith("_"):
            continue
        v = np.array([r[k] for r in rows], dtype=float)
        n = np.sum(~np.isnan(v))
        sd = np.nanstd(v, ddof=1) if n > 1 else 0.0
        ci = sps.t.ppf(0.975, n-1) * sd / np.sqrt(n) if n > 1 else 0.0
        out[k] = (float(np.nanmean(v)), float(sd), float(ci), v)
    return out


def paired_test(a, b, key="outage_rate"):

    x, y = a[key][3], b[key][3]
    d = x - y
    n = len(d)
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    ci = sps.t.ppf(0.975, n-1) * sd / np.sqrt(n) if n > 1 else 0.0
    p = float(sps.ttest_rel(x, y).pvalue) if n > 1 else float("nan")
    return m, ci, p


def fmt(agg, key, digits=3):
    m, sd, ci, _ = agg[key]
    return f"{m:.{digits}f} ± {ci:.{digits}f}"
''',
    "campaign": r'''import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import scipy

import experiment as E
from sim import Cfg

ROOT = Path(__file__).resolve().parents[1]
SIMULATION_SOURCES = ("sim.py", "planner.py", "baselines.py", "experiment.py", "campaign.py")


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def configurations():
    configs = {
        "ours": ("ours", {"lam": 2.0}, {}),
        "no_explore": ("ours", {"lam": 0.0}, {}),
        "oracle": ("oracle", {}, {}),
        "oblivious": ("oblivious", {}, {}),
        "twophase": ("twophase", {"te_steps": 60}, {}),
        "reactive": ("reactive", {}, {}),
        "static": ("static", {}, {}),
        "mean_only": ("ours", {"lam": 2.0, "use_chance": False}, {}),
        "wrong_kernel": ("ours", {"lam": 2.0, "ell_gp": 600.0}, {}),
        "online_hyper": ("ours", {"lam": 2.0, "online_hyper": True}, {}),
        "energy": ("ours", {"lam": 2.0, "w_energy": 0.5}, {}),
        "no_terminal": ("ours", {"lam": 2.0, "w_term": 0.0}, {}),
        "no_feedback": ("ours", {"lam": 2.0, "use_feedback": False}, {}),
        "no_acceleration": ("ours", {"lam": 2.0, "a_max": None}, {}),
    }
    for lam in (0.5, 1.0, 4.0, 8.0):
        configs[f"lambda_{lam:g}"] = ("ours", {"lam": lam}, {})
    for te in (0, 30, 120, 240):
        configs[f"te_{te}"] = ("twophase", {"te_steps": te}, {})
    for lookahead in (10, 30, 120):
        configs[f"lookahead_{lookahead}"] = (
            "ours", {"lam": 2.0, "lookahead": lookahead}, {}
        )
    for count in (50, 400):
        configs[f"nmax_{count}"] = ("ours", {"lam": 2.0, "max_obs": count}, {})
    for scale in (0.5, 2.0):
        for method in ("ours", "oracle", "oblivious"):
            configs[f"ell_{scale:g}_{method}"] = (
                method,
                {"lam": 2.0, "ell_gp": 250.0 * scale},
                {"SHADOW_CORR": 150.0 * scale},
            )
    for power in (20.0, 22.0, 26.0):
        for method in ("ours", "oblivious"):
            configs[f"pj_{power:g}_{method}"] = (
                method, {"lam": 2.0}, {"PJ_DBM": power}
            )
    for steps in (200, 800):
        for method in ("ours", "oracle"):
            configs[f"horizon_{steps}_{method}"] = (
                method, {"lam": 2.0}, {"T_STEPS": steps}
            )
    for method in ("ours", "oracle", "oblivious", "static"):
        configs[f"fading_{method}"] = (
            method, {"lam": 2.0, "use_fading": True}, {}
        )
    return configs


def source_hashes():
    return {
        name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
        for name in SIMULATION_SOURCES
    }


def signature(name, seed, method, kwargs, overrides, hashes):
    payload = {
        "config": name,
        "seed": seed,
        "method": method,
        "kwargs": kwargs,
        "overrides": overrides,
        "source_sha256": hashes,
    }
    raw = json.dumps(clean(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def completed(path, expected_signature):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text())["signature"] == expected_signature
    except (KeyError, OSError, json.JSONDecodeError):
        return False


def worker(job):
    name, seed, method, kwargs, overrides, output, run_signature = job
    cfg = type("EpisodeCfg", (Cfg,), overrides)
    started = time.time()
    result = E.run_episode(seed, method, cfg=cfg, record=True, **kwargs)
    series = {
        key: value
        for key, value in result.items()
        if key.startswith("_") and isinstance(value, np.ndarray)
    }
    metrics = {key: value for key, value in result.items() if not key.startswith("_")}
    target = output / "episodes" / name
    target.mkdir(parents=True, exist_ok=True)
    series_tmp = target / f"{seed}.series.tmp"
    with series_tmp.open("wb") as handle:
        np.savez_compressed(handle, **series)
    series_tmp.replace(target / f"{seed}.series.dat")
    record = {
        "signature": run_signature,
        "config": name,
        "seed": seed,
        "method": method,
        "kwargs": kwargs,
        "overrides": overrides,
        "wall_s": time.time() - started,
        "metrics": metrics,
    }
    metrics_tmp = target / f"{seed}.metrics.tmp"
    metrics_tmp.write_text(json.dumps(clean(record), indent=2, allow_nan=False))
    metrics_tmp.replace(target / f"{seed}.metrics.dat")
    return name, seed, record["wall_s"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the UAV relay experiment campaign.")
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--steps", type=int, help="Override mission length for every configuration.")
    parser.add_argument("--only", nargs="*", help="Run only named configurations.")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--force", action="store_true", help="Recompute matching completed jobs.")
    return parser.parse_args(argv)


def run(args):
    configs = configurations()
    if args.only:
        unknown = sorted(set(args.only) - set(configs))
        if unknown:
            raise SystemExit(f"Unknown configurations: {', '.join(unknown)}")
        configs = {name: configs[name] for name in args.only}
    if args.steps is not None:
        if args.steps < 2:
            raise SystemExit("--steps must be at least 2")
        configs = {
            name: (method, kwargs, {**overrides, "T_STEPS": args.steps})
            for name, (method, kwargs, overrides) in configs.items()
        }
    seeds = list(range(args.start, args.start + args.seeds))
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    jobs = []
    for name, config in configs.items():
        method, kwargs, overrides = config
        for seed in seeds:
            run_signature = signature(name, seed, method, kwargs, overrides, hashes)
            metric_path = output / "episodes" / name / f"{seed}.metrics.dat"
            series_path = output / "episodes" / name / f"{seed}.series.dat"
            if args.force or not (completed(metric_path, run_signature) and series_path.exists()):
                jobs.append((name, seed, method, kwargs, overrides, output, run_signature))
    manifest = {
        "seeds": seeds,
        "configs": configs,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "workers": args.workers,
        "steps_override": args.steps,
        "source_sha256": hashes,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "manifest.dat").write_text(
        json.dumps(clean(manifest), indent=2, allow_nan=False)
    )
    print(f"configs={len(configs)} seeds={len(seeds)} remaining={len(jobs)}", flush=True)
    started = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for index, future in enumerate(cf.as_completed(futures), 1):
            name, seed, seconds = future.result()
            elapsed = (time.time() - started) / 60.0
            print(
                f"{index}/{len(jobs)} {name} seed={seed} {seconds:.1f}s elapsed={elapsed:.1f}min",
                flush=True,
            )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    run(parse_args())
''',
    "parametric_baseline": r'''import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from scipy.optimize import least_squares

import experiment
from campaign import clean, source_hashes
from sim import Cfg

ROOT = Path(__file__).resolve().parents[1]


def signature(seed):
    payload = {"seed": seed, "source_sha256": source_hashes(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class JammerField:
    def __init__(self, ell=250.0, sigma_f=12.0, sigma_y=1.0,
                 mean_dbm=-105.0, max_obs=150):
        self.ell, self.sf, self.sy = ell, sigma_f, sigma_y
        self.m0, self.max_obs = mean_dbm, max_obs
        self.X = np.empty((0, 3))
        self.y = np.empty(0)
        self.theta = None
        self.total_obs = 0
        self.last_fit_count = 0
        self.residual_sd = 12.0

    def add(self, x, y):
        self.X = np.vstack((self.X, np.asarray(x)[None, :]))[-self.max_obs:]
        self.y = np.append(self.y, y)[-self.max_obs:]
        self.total_obs += 1

    def _model(self, theta, positions):
        distance = np.sqrt(np.sum((positions[:, :2]-theta[:2])**2, axis=1)
                           + positions[:, 2]**2)
        return theta[2]-30.0*np.log10(np.maximum(distance, 1.0))

    def _fit(self):
        n = len(self.y)
        if n < 6 or (self.theta is not None and self.total_obs-self.last_fit_count < 10):
            return
        strongest = self.X[int(np.argmax(self.y)), :2]
        initial = [np.array([strongest[0], strongest[1], -16.0])]
        if self.theta is not None:
            initial.insert(0, self.theta.copy())
        else:
            initial.append(np.array([Cfg.L/2, Cfg.L/2, -16.0]))
        best = None
        for start in initial:
            result = least_squares(lambda theta: (self._model(theta, self.X)-self.y)/6.0,
                                   start, bounds=([0.0,0.0,-50.0],[Cfg.L,Cfg.L,30.0]),
                                   loss="soft_l1", max_nfev=50)
            score = np.mean((self._model(result.x, self.X)-self.y)**2)
            if best is None or score < best[0]:
                best = (score, result.x)
        self.theta = best[1]
        self.residual_sd = float(np.clip(np.sqrt(best[0]), 3.0, 15.0))
        self.last_fit_count = self.total_obs

    def predict(self, points):
        points = np.atleast_2d(points)
        self._fit()
        if self.theta is None:
            return np.full(len(points), self.m0), np.full(len(points), self.sf)
        mean = self._model(self.theta, points)
        sd = np.full(len(points), self.residual_sd)
        return mean, sd

    def argmax_grid(self, n=41, L=None):
        self._fit()
        return np.array([Cfg.L/2, Cfg.L/2]) if self.theta is None else self.theta[:2].copy()


def run_one(seed):
    experiment.GaussianField = JammerField
    output = ROOT / "results_round2_parametric" / "episodes" / "parametric"
    target = output / f"{seed}.metrics.dat"
    run_signature = signature(seed)
    if target.exists() and (output / f"{seed}.series.dat").exists():
        try:
            if json.loads(target.read_text())["signature"] == run_signature:
                return seed
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    started = time.time()
    result = experiment.run_episode(seed, "ours", lam=0.0, record=True)
    output.mkdir(parents=True, exist_ok=True)
    series = {key: value for key,value in result.items()
              if key.startswith("_") and isinstance(value, np.ndarray)}
    with (output / f"{seed}.series.tmp").open("wb") as handle:
        np.savez_compressed(handle, **series)
    (output / f"{seed}.series.tmp").replace(output / f"{seed}.series.dat")
    record = {"seed": seed, "method": "parametric-jammer-sampler",
              "signature": run_signature,
              "wall_s": time.time()-started,
              "metrics": {key: value for key,value in result.items() if not key.startswith("_")}}
    (output / f"{seed}.metrics.dat").write_text(json.dumps(clean(record), indent=2, allow_nan=False))
    return seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output = ROOT / "results_round2_parametric"
    output.mkdir(exist_ok=True)
    metadata = {"seed_start": args.start, "seeds": args.seeds,
                "jammer_pathloss_exponent": 3.0, "fit_frequency_observations": 10,
                "source_sha256": source_hashes(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "manifest.dat").write_text(json.dumps(metadata, indent=2))
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, seed in enumerate(pool.map(run_one, range(args.start,args.start+args.seeds)),1):
            if index % 10 == 0 or index == args.seeds:
                print(f"{index}/{args.seeds} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
''',
    "round2_mechanism": r'''import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats

import experiment
from campaign import source_hashes
from planner import MPPIPlanner, outage_prob
from sim import Cfg, prop_power, hover_power

ROOT = Path(__file__).resolve().parents[1]


class TracedPlanner(MPPIPlanner):
    current = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []
        TracedPlanner.current = self

    def step(self, q, gp, g_now, g_vel):
        cfg, H, K = self.cfg, self.H, self.K
        rng = self.rng
        g_pred = g_now + np.outer(np.arange(1, H + 1) * cfg.DT, g_vel)
        noise = rng.normal(0.0, self.sigma_v, size=(K, H, 2))
        U = self.u_prev[None] + noise
        if self.a_max is not None:
            lim = self.a_max * cfg.DT
            prev = np.tile(self.last_velocity, (K, 1))
            for h in range(H):
                dv = U[:, h] - prev
                dn = np.linalg.norm(dv, axis=-1, keepdims=True)
                dv = np.where(dn > lim, dv / np.maximum(dn, 1e-9) * lim, dv)
                U[:, h] = prev + dv
                prev = U[:, h]
        nrm = np.linalg.norm(U, axis=-1, keepdims=True)
        U = np.where(nrm > cfg.V_MAX, U / np.maximum(nrm, 1e-9) * cfg.V_MAX, U)
        Q = q[None, None, :] + np.cumsum(U * cfg.DT, axis=1)
        Q = np.clip(Q, 0.0, cfg.L)
        flatQ = Q.reshape(-1, 2)
        mu_q, sd_q = self._predict(gp, np.column_stack([flatQ, np.full(len(flatQ), cfg.H_UAV)]))
        mu_g, sd_g = self._predict(gp, np.column_stack([g_pred, np.zeros(len(g_pred))]))
        mu_g = np.tile(mu_g, K)
        sd_g = np.tile(sd_g, K)
        G = np.tile(g_pred, (K, 1))
        p_out = outage_prob(flatQ, G, mu_q, sd_q, mu_g, sd_g, cfg,
                            use_chance=self.use_chance)
        g_far = g_now + np.outer(np.arange(1, self.lookahead + 1) * cfg.DT, g_vel)
        g_far = np.clip(g_far, 0.0, cfg.L)
        d_far = cdist(flatQ, g_far)
        relevance = np.exp(-(d_far.min(axis=1) / (1.5 * gp.ell)) ** 2)
        info_gain = (sd_q / gp.sf) * relevance
        if self.w_energy > 0.0:
            speeds = np.linalg.norm(U, axis=-1).reshape(-1)
            c_energy = prop_power(speeds, cfg) / hover_power(cfg)
        else:
            c_energy = 0.0
        run_cost = (p_out - self.lam * info_gain
                    + self.w_energy * c_energy).reshape(K, H).sum(axis=1)
        if self.w_term > 0.0:
            step = max(1, len(g_far) // self.n_term)
            gf = g_far[::step][:self.n_term]
            Fs = len(gf)
            qT = Q[:, -1, :]
            mu_T, sd_T = self._predict(gp, np.column_stack([qT, np.full(len(qT), cfg.H_UAV)]))
            mu_f, sd_f = self._predict(gp, np.column_stack([gf, np.zeros(len(gf))]))
            QT = np.repeat(qT, Fs, axis=0)
            GF = np.tile(gf, (K, 1))
            p_term = outage_prob(QT, GF,
                                 np.repeat(mu_T, Fs), np.repeat(sd_T, Fs),
                                 np.tile(mu_f, K), np.tile(sd_f, K),
                                 cfg, use_chance=self.use_chance)
            cost = run_cost + self.w_term * p_term.reshape(K, Fs).mean(axis=1)
        else:
            cost = run_cost
        w = np.exp(-(cost - cost.min()) / self.temp)
        w /= w.sum()
        self.trace.append((float(1.0 / np.sum(w ** 2) / K),
                           float(np.mean(p_out > .99)),
                           float(np.std(cost))))
        u_new = np.einsum("k,khd->hd", w, U)
        self.u_prev = np.vstack([u_new[1:], u_new[-1:]])
        return u_new[0]


def evaluate(job):
    seed, power = job
    original = experiment.MPPIPlanner
    experiment.MPPIPlanner = TracedPlanner
    try:
        cfg = type("TraceCfg", (Cfg,), {"PJ_DBM": power})
        result = experiment.run_episode(seed, "ours", lam=0.0, cfg=cfg, record=True)
        trace = np.asarray(TracedPlanner.current.trace)
    finally:
        experiment.MPPIPlanner = original
    failures = result["_outages"].astype(bool)
    assert len(trace) == len(failures)
    if power == 24.0:
        with np.load(ROOT / "results" / "episodes" / "no_explore" / f"{seed}.series.dat") as archive:
            if not np.allclose(result["_traj"], archive["_traj"], atol=1e-8):
                raise RuntimeError(f"trace changed path for seed {seed}")
    return {"seed": seed, "jammer_dbm": power,
            "outage_rate": result["outage_rate"],
            "ess_fraction_mean": float(trace[:, 0].mean()),
            "ess_fraction_failed": float(trace[failures, 0].mean()) if failures.any() else None,
            "ess_fraction_served": float(trace[~failures, 0].mean()) if (~failures).any() else None,
            "risk_saturation_failed": float(trace[failures, 1].mean()) if failures.any() else None,
            "risk_saturation_served": float(trace[~failures, 1].mean()) if (~failures).any() else None,
            "cost_sd_failed": float(trace[failures, 2].mean()) if failures.any() else None}


def main():
    jobs = [(seed, power) for power in (24.0, 32.0) for seed in range(1000, 1030)]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(evaluate, jobs):
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"{len(rows)}/{len(jobs)}", flush=True)
    summary = {}
    for power in (24.0, 32.0):
        subset = [row for row in rows if row["jammer_dbm"] == power]
        summary[str(power)] = {key: float(np.mean([row[key] for row in subset if row[key] is not None]))
                               for key in ("outage_rate", "ess_fraction_mean",
                                           "ess_fraction_failed", "ess_fraction_served",
                                           "risk_saturation_failed", "risk_saturation_served",
                                           "cost_sd_failed")}
    output = {"source_sha256": source_hashes(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summary": summary, "per_seed": rows}
    (ROOT / "results_round2_mechanism.dat").write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
''',
    "round3_campaign": r'''import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np

import experiment
from campaign import clean, source_hashes
from round2_mechanism import TracedPlanner
from sim import Cfg

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_round3"
SEEDS = range(2000, 2200)
TEMPERATURES = (0.05, 0.1, 0.3, 1.0)


def settings():
    rows = [(name, method, 1.0, power) for name, method, power in (
        ("power_30_gp", "ours", 30.0),
        ("power_30_geometry", "oblivious", 30.0),
        ("power_32_gp", "ours", 32.0),
    )]
    rows.extend((f"tau_{tau:g}_{method}", method, tau, 24.0)
                for tau in (1.0, 0.3, 0.1, 0.05) for method in ("ours", "oracle"))
    return rows


class DiagnosticPlanner(TracedPlanner):
    selected_temperature = 1.0

    def __init__(self, *args, **kwargs):
        kwargs["temperature"] = self.selected_temperature
        super().__init__(*args, **kwargs)
        self.extra = []

    def step(self, q, gp, g_now, g_vel):
        state = self.rng.bit_generator.state
        probe = np.random.default_rng()
        probe.bit_generator.state = state
        noise = probe.normal(0.0, self.sigma_v, size=(self.K, self.H, 2))
        initial = self.u_prev[0] + noise[:, 0]
        change = np.linalg.norm(initial - self.last_velocity, axis=1)
        limit = self.a_max * self.cfg.DT if self.a_max is not None else np.inf
        clipped = float(np.mean(change > limit))
        reference = self.u_prev[0].copy()
        prior_velocity = self.last_velocity.copy()
        velocity = super().step(q, gp, g_now, g_vel)
        target = 0.5 * (self.cfg.BS[:2] + g_now)
        displacement = target - q
        toward = np.dot(velocity, displacement) / max(np.linalg.norm(displacement), 1e-9)
        self.extra.append((clipped, float(np.linalg.norm(velocity-reference)),
                           float(np.linalg.norm(velocity-prior_velocity)),
                           float(toward)))
        return velocity


def source_signature():
    payload = {"sources": source_hashes(),
               "round2_trace": hashlib.sha256((ROOT / "src/round2_mechanism.py").read_bytes()).hexdigest(),
               "round3_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate(job):
    name, method, temperature, power, seed, signature = job
    target = OUTPUT / "episodes" / name
    metrics_path = target / f"{seed}.metrics.dat"
    series_path = target / f"{seed}.series.dat"
    if metrics_path.exists() and series_path.exists():
        try:
            if json.loads(metrics_path.read_text())["signature"] == signature:
                return name, seed, "cached"
        except (KeyError, ValueError):
            pass
    cfg = type("Round3Cfg", (Cfg,), {"PJ_DBM": power})
    original = experiment.MPPIPlanner
    DiagnosticPlanner.selected_temperature = temperature
    experiment.MPPIPlanner = DiagnosticPlanner
    try:
        result = experiment.run_episode(seed, method, lam=0.0, cfg=cfg, record=True)
        planner = TracedPlanner.current
        trace = np.asarray(planner.trace, float).reshape(-1, 3)
        extra = np.asarray(planner.extra, float).reshape(-1, 4)
    finally:
        experiment.MPPIPlanner = original
    failed = np.asarray(result["_outages"], bool)
    if method == "oblivious":
        assert len(trace) == 0
    else:
        assert len(trace) == len(failed) == len(extra) == cfg.T_STEPS
    if power == 24.0 and temperature == 1.0:
        folder = "results_validation" if method == "ours" else "results_round2_validation"
        key = "no_explore" if method == "ours" else "oracle"
        old = json.loads((ROOT / folder / "episodes" / key / f"{seed}.metrics.dat").read_text())
        if abs(result["outage_rate"] - old["metrics"]["outage_rate"]) > 1e-12:
            raise RuntimeError(f"baseline replay mismatch: {name} seed {seed}")
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature, "config": name, "seed": seed,
        "method": method, "temperature": temperature, "jammer_dbm": power,
        "metrics": {key: value for key, value in result.items() if not key.startswith("_")},
    }
    tmp = metrics_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(payload), allow_nan=False, indent=2))
    tmp.replace(metrics_path)
    tmp = series_path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, outages=failed, trace=trace, extra=extra,
                            trajectory=result["_traj"])
    tmp.replace(series_path)
    return name, seed, "new"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    signature = source_signature()
    OUTPUT.mkdir(exist_ok=True)
    manifest = {"seeds": [2000, 2199], "temperatures": TEMPERATURES,
                "settings": settings(), "signature": signature,
                "source_sha256": source_hashes()}
    (OUTPUT / "manifest.dat").write_text(json.dumps(clean(manifest), indent=2))
    jobs = [(name, method, tau, power, seed, signature)
            for name, method, tau, power in settings() for seed in SEEDS]
    started = time.time()
    print(f"jobs={len(jobs)} workers={args.workers}", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for index, future in enumerate(cf.as_completed(futures), 1):
            name, seed, state = future.result()
            if index % 20 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} {name} {seed} {state} "
                      f"elapsed_min={(time.time()-started)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
''',
    "graph_search_baseline": r'''import numpy as np

from planner import outage_prob
from sim import Cfg, TrueField, make_ground_track


class LatticePlanner:
    def __init__(self, lam=0.0, horizon=10, n_samples=150, temperature=1.0,
                 sigma_v=8.0, oracle_fn=None, cfg=Cfg, lookahead=60,
                 seed=0, w_energy=0.0, use_chance=True, a_max=None,
                 w_term=None, n_term=10, beam_width=16, depth=6):
        self.cfg = cfg
        self.oracle_fn = oracle_fn
        self.a_max = cfg.A_MAX if a_max is None else a_max
        self.beam_width = beam_width
        self.depth = depth
        self.last_velocity = np.zeros(2)
        angles = np.arange(8) * (np.pi / 4)
        # 一个零加速度动作加八个水平加速方向。
        self.actions = np.vstack((np.zeros(2),
                                  self.a_max * cfg.DT *
                                  np.column_stack((np.cos(angles), np.sin(angles)))))

    def _predict(self, gp, positions):
        if self.oracle_fn is None:
            return gp.predict(positions)
        return self.oracle_fn(positions), np.full(len(positions), gp.sy)

    def step(self, q, gp, g_now, g_vel):
        cfg = self.cfg
        positions = np.asarray(q, float)[None, :]
        velocities = np.asarray(self.last_velocity, float)[None, :]
        first = np.zeros((1, 2))
        costs = np.zeros(1)
        for h in range(self.depth):
            count = len(positions)
            velocity = velocities[:, None, :] + self.actions[None, :, :]
            speed = np.linalg.norm(velocity, axis=2, keepdims=True)
            velocity = np.where(speed > cfg.V_MAX,
                                velocity * cfg.V_MAX / np.maximum(speed, 1e-9),
                                velocity)
            velocity = velocity.reshape(-1, 2)
            position = np.clip(np.repeat(positions, len(self.actions), axis=0)
                               + velocity * cfg.DT, 0.0, cfg.L)
            velocity = (position - np.repeat(positions, len(self.actions), axis=0)) / cfg.DT
            user = np.clip(g_now + (h + 1) * cfg.DT * g_vel, 0.0, cfg.L)
            relay_points = np.column_stack((position, np.full(len(position), cfg.H_UAV)))
            user_point = np.r_[user, 0.0][None, :]
            mu_q, sd_q = self._predict(gp, relay_points)
            mu_g, sd_g = self._predict(gp, user_point)
            risk = outage_prob(position, np.repeat(user[None, :], len(position), axis=0),
                               mu_q, sd_q, np.full(len(position), mu_g[0]),
                               np.full(len(position), sd_g[0]), cfg,
                               use_chance=False)
            midpoint = 0.5 * (cfg.BS[:2] + user)
            distance = np.linalg.norm(position - midpoint, axis=1) / cfg.L
            cost = np.repeat(costs, len(self.actions)) + risk + 0.05 * distance
            first_expanded = np.repeat(first, len(self.actions), axis=0)
            if h == 0:
                first_expanded = velocity.copy()
            # 每层保留部分路径，同时保存它们对应的首步速度。
            keep = np.argsort(cost, kind="stable")[:self.beam_width]
            positions, velocities, first, costs = (position[keep], velocity[keep],
                                                   first_expanded[keep], cost[keep])
        return first[int(np.argmin(costs))]


class ExactOraclePlanner(LatticePlanner):
    # 仅对短时离散动作枚举精确，不代表整段飞行的全局最优。
    def __init__(self, *args, seed=0, depth=3, **kwargs):
        super().__init__(*args, seed=seed, depth=depth, **kwargs)
        if self.oracle_fn is None:
            raise ValueError("the exact lattice oracle needs the true field")
        rng = np.random.default_rng(seed)
        TrueField(rng, self.cfg)
        self.user_track = make_ground_track(rng, self.cfg)
        self.slot = 0

    def step(self, q, gp, g_now, g_vel):
        cfg = self.cfg
        positions = np.asarray(q, float)[None, :]
        velocities = np.asarray(self.last_velocity, float)[None, :]
        first = np.zeros((1, 2))
        costs = np.zeros(1)
        for h in range(self.depth):
            velocity = velocities[:, None, :] + self.actions[None, :, :]
            speed = np.linalg.norm(velocity, axis=2, keepdims=True)
            velocity = np.where(speed > cfg.V_MAX,
                                velocity * cfg.V_MAX / np.maximum(speed, 1e-9),
                                velocity).reshape(-1, 2)
            previous = np.repeat(positions, len(self.actions), axis=0)
            position = np.clip(previous + velocity * cfg.DT, 0.0, cfg.L)
            velocity = (position - previous) / cfg.DT
            user = self.user_track[min(self.slot+h, len(self.user_track)-1)]
            relay_points = np.column_stack((position, np.full(len(position), cfg.H_UAV)))
            mu_q = self.oracle_fn(relay_points)
            mu_g = float(self.oracle_fn(np.r_[user, 0.0][None, :])[0])
            risk = outage_prob(position, np.repeat(user[None, :], len(position), axis=0),
                               mu_q, np.ones(len(position)),
                               np.full(len(position), mu_g), np.ones(len(position)),
                               cfg, use_chance=False)
            midpoint = 0.5 * (cfg.BS[:2] + user)
            distance = np.linalg.norm(position - midpoint, axis=1) / cfg.L
            costs = np.repeat(costs, len(self.actions)) + risk + 0.05 * distance
            first = np.repeat(first, len(self.actions), axis=0)
            if h == 0:
                first = velocity.copy()
            positions, velocities = position, velocity
        self.slot += 1
        return first[int(np.argmin(costs))]
''',
    "round4_campaign": r'''import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np

import experiment
from campaign import clean, source_hashes
from graph_search_baseline import LatticePlanner, ExactOraclePlanner
from parametric_baseline import JammerField
from planner import MPPIPlanner, GaussianField
from round3_campaign import DiagnosticPlanner
from round2_mechanism import TracedPlanner
from sim import Cfg, TrueField

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_round4"
DEVELOPMENT = range(3000, 3100)
HOLDOUT = range(4000, 4200)
RATE = 3e6
GRID = ((0.05, 4.0), (0.05, 8.0), (0.1, 4.0),
        (0.1, 8.0), (0.3, 4.0), (0.3, 8.0))
EXTRA_GRID = ((0.05, 2.0), (0.1, 2.0))
GRAPH_GRID = ((16, 6), (32, 8))
CONFIRM = ("gp_tuned", "geometry", "known_tuned", "graph_gp",
           "graph_known", "oracle_lattice_exact", "gp_nominal", "corridor_tuned", "parametric_tuned",
           "reactive", "twophase_tuned")
STRESS = ("gp_tuned", "geometry", "graph_gp", "known_tuned")


def digest():
    # 签名覆盖实验配置和仿真源码，防止断点续跑混入旧结果。
    files = [Path(__file__), ROOT / "src/graph_search_baseline.py",
             ROOT / "src/parametric_baseline.py",
             ROOT / "src/round3_campaign.py",
             ROOT / "src/round2_mechanism.py"]
    payload = {"core": source_hashes(),
               "new": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               "rate": RATE, "grid": GRID, "extra_grid": EXTRA_GRID,
               "graph_grid": GRAPH_GRID}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parameters(name, selected):
    if name.startswith("cal_tau"):
        parts = name.split("_")
        return float(parts[1][3:]), float(parts[2][5:])
    if name == "gp_nominal":
        return 1.0, 8.0
    if name.startswith("graph_b"):
        return 1.0, 8.0
    return selected["temperature"], selected["sigma_v"]


def evaluate(job):
    phase, name, seed, selected, signature = job
    target = OUTPUT / phase / name
    metrics_path = target / f"{seed}.metrics.dat"
    series_path = target / f"{seed}.series.dat"
    # 两类输出都存在且签名一致时才复用该场景。
    if metrics_path.exists() and series_path.exists():
        try:
            if json.loads(metrics_path.read_text())["signature"] == signature:
                return phase, name, seed, "cached"
        except (ValueError, KeyError):
            pass
    tau, sigma = parameters(name, selected)
    rate = 4.5e6 if phase == "stress" else RATE
    cfg = type("StudyCfg", (Cfg,), {"R_MIN": rate})
    original_planner = experiment.MPPIPlanner
    original_map = experiment.GaussianField

    class ConfiguredPlanner(DiagnosticPlanner):
        selected_temperature = tau

        def __init__(self, *args, **kwargs):
            kwargs["sigma_v"] = sigma
            super().__init__(*args, **kwargs)

    if name.startswith("graph_b"):
        beam = int(name.split("_")[1][1:])
        depth = int(name.split("_")[2][1:])
    elif name.startswith("graph_"):
        beam, depth = selected["graph_beam"], selected["graph_depth"]
    else:
        beam, depth = 16, 6

    class ConfiguredLattice(LatticePlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, beam_width=beam, depth=depth, **kwargs)

    if name == "oracle_lattice_exact":
        experiment.MPPIPlanner = ExactOraclePlanner
    else:
        experiment.MPPIPlanner = ConfiguredLattice if name.startswith("graph_") else ConfiguredPlanner
    if name == "parametric_tuned":
        experiment.GaussianField = JammerField
    if name == "geometry":
        method = "oblivious"
    elif name == "reactive":
        method = "reactive"
    elif name in ("known_tuned", "graph_known", "oracle_lattice_exact"):
        method = "oracle"
    elif name == "twophase_tuned":
        method = "twophase"
    else:
        method = "ours"
    lam = 2.0 if name == "corridor_tuned" else 0.0
    started = time.time()
    try:
        result = experiment.run_episode(seed, method, lam=lam, cfg=cfg, record=True)
        if experiment.MPPIPlanner is ConfiguredPlanner:
            trace = np.asarray(TracedPlanner.current.trace, float)
            extra = np.asarray(TracedPlanner.current.extra, float)
        else:
            trace = np.empty((0, 3))
            extra = np.empty((0, 4))
    finally:
        experiment.MPPIPlanner = original_planner
        experiment.GaussianField = original_map
    target.mkdir(parents=True, exist_ok=True)
    scene = TrueField(np.random.default_rng(seed), cfg)
    record = {"signature": signature, "phase": phase, "name": name,
              "seed": seed, "rate_bps": rate, "temperature": tau,
              "sigma_v": sigma, "jammer_xy": scene.p_J.tolist(),
              "wall_s": time.time() - started,
              "metrics": {key: value for key, value in result.items()
                          if not key.startswith("_")}}
    tmp = metrics_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(record), allow_nan=False, indent=2))
    tmp.replace(metrics_path)
    tmp = series_path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, rates=result["_rates"],
                            outages=result["_outages"],
                            trajectory=result["_traj"],
                            user=result["_gtraj"], trace=trace, extra=extra)
    tmp.replace(series_path)
    return phase, name, seed, "new"


def selected_parameters():
    path = OUTPUT / "selection.dat"
    if not path.exists():
        raise RuntimeError("run round4_analysis.py select before confirm")
    record = json.loads(path.read_text())
    if record["signature"] != digest():
        raise RuntimeError("selection signature does not match current sources")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibrate", "calibrate_extra", "graph_calibrate", "audit", "confirm", "stress"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    signature = digest()
    OUTPUT.mkdir(exist_ok=True)
    manifest = {"signature": signature, "rate_bps": RATE,
                "development_seeds": [3000, 3099],
                "holdout_seeds": [4000, 4199], "grid": GRID,
                "extra_grid": EXTRA_GRID,
                "graph_grid": GRAPH_GRID,
                "confirm_methods": CONFIRM,
                "stress_methods": STRESS,
                "selection_rule": "lowest development mean outage; tie by 90th percentile longest outage; then lower temperature and spread",
                "source_sha256": source_hashes()}
    (OUTPUT / "manifest.dat").write_text(json.dumps(clean(manifest), indent=2))
    selected = selected_parameters() if args.phase in ("audit", "confirm", "stress") else None
    if args.phase == "calibrate":
        names = [f"cal_tau{tau:g}_sigma{sigma:g}" for tau, sigma in GRID]
        seeds = DEVELOPMENT
    elif args.phase == "calibrate_extra":
        names = [f"cal_tau{tau:g}_sigma{sigma:g}" for tau, sigma in EXTRA_GRID]
        seeds = DEVELOPMENT
    elif args.phase == "graph_calibrate":
        names = [f"graph_b{beam}_h{depth}" for beam, depth in GRAPH_GRID]
        seeds = DEVELOPMENT
    elif args.phase == "audit":
        names, seeds = ("corridor_tuned", "geometry"), DEVELOPMENT
    elif args.phase == "stress":
        names, seeds = STRESS, HOLDOUT
    else:
        names, seeds = CONFIRM, HOLDOUT
    jobs = [(args.phase, name, seed, selected, signature)
            for name in names for seed in seeds]
    start = time.time()
    print(f"phase={args.phase} jobs={len(jobs)} workers={args.workers}", flush=True)
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(evaluate, job) for job in jobs]
        for index, future in enumerate(futures.as_completed(pending), 1):
            phase, name, seed, state = future.result()
            if index % 20 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} {name} {seed} {state} "
                      f"elapsed_min={(time.time()-start)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
''',
    "round4_analysis": r'''import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from round4_campaign import (OUTPUT, DEVELOPMENT, HOLDOUT, GRID, EXTRA_GRID, GRAPH_GRID,
                             CONFIRM, STRESS, digest)


def load(phase, name, seeds):
    records = []
    for seed in seeds:
        path = OUTPUT / phase / name / f"{seed}.metrics.dat"
        record = json.loads(path.read_text())
        if record["seed"] != seed:
            raise ValueError(path)
        records.append(record)
    if len({r["signature"] for r in records}) != 1:
        raise ValueError(f"mixed source signatures in {phase}/{name}")
    return records


def mean_ci(values):
    values = np.asarray(values, float)
    center = float(np.mean(values))
    radius = float(stats.t.ppf(.975, len(values)-1) *
                   np.std(values, ddof=1) / np.sqrt(len(values)))
    return {"mean": center, "ci": [center-radius, center+radius]}


def select():
    rows = []
    for tau, sigma in (*GRID, *EXTRA_GRID):
        name = f"cal_tau{tau:g}_sigma{sigma:g}"
        phase = "calibrate_extra" if (tau, sigma) in EXTRA_GRID else "calibrate"
        records = load(phase, name, DEVELOPMENT)
        outages = np.array([r["metrics"]["outage_rate"] for r in records])
        longest = np.array([r["metrics"]["longest_outage_s"] for r in records])
        rows.append({"name": name, "temperature": tau, "sigma_v": sigma,
                     "outage": mean_ci(outages),
                     "longest_p90": float(np.percentile(longest, 90)),
                     "calibration_signature": records[0]["signature"]})
    rows.sort(key=lambda row: (row["outage"]["mean"], row["longest_p90"],
                               row["temperature"], row["sigma_v"]))
    graph_rows = []
    for beam, depth in GRAPH_GRID:
        name = f"graph_b{beam}_h{depth}"
        records = load("graph_calibrate", name, DEVELOPMENT)
        outages = np.array([r["metrics"]["outage_rate"] for r in records])
        longest = np.array([r["metrics"]["longest_outage_s"] for r in records])
        graph_rows.append({"name": name, "beam": beam, "depth": depth,
                           "outage": mean_ci(outages),
                           "longest_p90": float(np.percentile(longest, 90)),
                           "calibration_signature": records[0]["signature"]})
    graph_rows.sort(key=lambda row: (row["outage"]["mean"], row["longest_p90"],
                                     row["beam"], row["depth"]))
    selected = {"signature": digest(),
                "calibration_signature": rows[0]["calibration_signature"],
                "temperature": rows[0]["temperature"],
                "sigma_v": rows[0]["sigma_v"],
                "graph_beam": graph_rows[0]["beam"],
                "graph_depth": graph_rows[0]["depth"],
                "selected_from": [3000, 3099],
                "holdout": [4000, 4199],
                "selection_rule": "lowest development mean outage; tie by 90th percentile longest outage; then lower temperature and spread",
                "ranking": rows, "graph_ranking": graph_rows}
    (OUTPUT / "selection.dat").write_text(json.dumps(selected, indent=2))
    print(json.dumps(selected, indent=2))


def pair(left, right):
    delta = np.asarray(left, float) - np.asarray(right, float)
    estimate = mean_ci(delta)
    estimate["p_t_two_sided"] = float(stats.ttest_1samp(delta, 0).pvalue)
    rng = np.random.default_rng(94131)
    extreme = 0
    observed = abs(np.mean(delta))
    # 符号翻转以场景差值为单位，不把 400 个时隙当独立样本。
    for _ in range(10):
        signs = rng.choice((-1.0, 1.0), size=(2000, len(delta)))
        extreme += int(np.sum(np.abs((signs * delta).mean(axis=1)) >= observed-1e-12))
    estimate["p_permutation"] = (extreme+1)/20001
    estimate["wins_ties_losses"] = [int(np.sum(delta < 0)),
                                    int(np.sum(delta == 0)),
                                    int(np.sum(delta > 0))]
    return estimate


def holm(p_values):
    names = sorted(p_values, key=p_values.get)
    adjusted = {}
    previous = 0.0
    # 按原始 p 值排序，并保持调整后的结果单调。
    for position, name in enumerate(names):
        value = min(1.0, p_values[name] * (len(names)-position))
        previous = max(previous, value)
        adjusted[name] = previous
    return adjusted


def summarize():
    records = {name: load("confirm", name, HOLDOUT) for name in CONFIRM}
    metrics = {}
    for name, rows in records.items():
        metrics[name] = {key: mean_ci([row["metrics"][key] for row in rows])
                         for key in ("outage_rate", "longest_outage_s", "rate_p05",
                                     "energy_kj", "solve_ms_med")}
    arrays = {name: [r["metrics"]["outage_rate"] for r in rows]
              for name, rows in records.items()}
    paired = {f"gp_minus_{name}": pair(arrays["gp_tuned"], arrays[name])
              for name in CONFIRM if name != "gp_tuned"}
    family = ("geometry", "graph_gp", "reactive", "corridor_tuned",
              "parametric_tuned")
    adjusted = holm({name: paired[f"gp_minus_{name}"]["p_permutation"]
                     for name in family})
    for name, value in adjusted.items():
        paired[f"gp_minus_{name}"]["p_holm"] = value
    traces = {}
    for name in ("gp_tuned", "gp_nominal", "known_tuned"):
        ess, clip, ref_change, offsets = [], [], [], []
        for seed in HOLDOUT:
            path = OUTPUT / "confirm" / name / f"{seed}.series.dat"
            with np.load(path) as series:
                trace = series["trace"]
                extra = series["extra"]
                q = series["trajectory"]
                g = series["user"][:len(q)]
            if len(trace) != len(q) or len(extra) != len(q):
                raise ValueError(f"incomplete sampler trace: {path}")
            ess.append(trace[:, 0])
            clip.append(extra[:, 0])
            ref_change.append(extra[:, 1])
            midpoint = 0.5 * (np.array([1000.0, 1000.0]) + g)
            offsets.append(np.linalg.norm(q-midpoint, axis=1))
        ess = np.concatenate(ess)
        offsets = np.concatenate(offsets)
        traces[name] = {"all_slot_ess_mean": float(np.mean(ess)),
                        "all_slot_ess_median": float(np.median(ess)),
                        "all_slot_ess_over_0.9": float(np.mean(ess > .9)),
                        "first_step_clip_fraction": float(np.mean(np.concatenate(clip))),
                        "chosen_vs_reference_mps": float(np.mean(np.concatenate(ref_change))),
                        "offset_over_500_fraction": float(np.mean(offsets > 500))}
    selected = json.loads((OUTPUT / "selection.dat").read_text())
    dev_name = f"cal_tau{selected['temperature']:g}_sigma{selected['sigma_v']:g}"
    development_phase = "calibrate_extra" if (selected["temperature"], selected["sigma_v"]) in EXTRA_GRID else "calibrate"
    development = load(development_phase, dev_name, DEVELOPMENT)
    dev_corridor = load("audit", "corridor_tuned", DEVELOPMENT)
    dev_geometry = load("audit", "geometry", DEVELOPMENT)
    dev_x = np.array([r["jammer_xy"][0] for r in development])
    hold_x = np.array([r["jammer_xy"][0] for r in records["gp_tuned"]])
    dev_effect = np.array([a["metrics"]["outage_rate"]-b["metrics"]["outage_rate"]
                           for a, b in zip(development, dev_corridor)])
    hold_effect = np.array([a-b for a,b in zip(arrays["gp_tuned"], arrays["corridor_tuned"])])
    audit = {"jammer_x_mean_development": float(dev_x.mean()),
             "jammer_x_mean_holdout": float(hold_x.mean()),
             "jammer_x_ks_p": float(stats.ks_2samp(dev_x, hold_x).pvalue),
             "paired_corridor_effect_development": mean_ci(dev_effect),
             "paired_corridor_effect_holdout": mean_ci(hold_effect),
             "paired_effect_ks_p": float(stats.ks_2samp(dev_effect, hold_effect).pvalue),
             "development_gp_minus_geometry": pair(
                 [r["metrics"]["outage_rate"] for r in development],
                 [r["metrics"]["outage_rate"] for r in dev_geometry])}
    stress_records = {name: load("stress", name, HOLDOUT) for name in STRESS}
    stress_outage = {name: mean_ci([r["metrics"]["outage_rate"] for r in rows])
                     for name, rows in stress_records.items()}
    stress_paired = {f"gp_minus_{name}": pair(
                         [r["metrics"]["outage_rate"] for r in stress_records["gp_tuned"]],
                         [r["metrics"]["outage_rate"] for r in stress_records[name]])
                     for name in STRESS if name != "gp_tuned"}
    output = {"selected": selected, "methods": metrics,
              "paired": paired, "holm_family": family, "batch_audit": audit,
              "trace_diagnostics": traces,
              "high_load": {"outage": stress_outage, "paired": stress_paired}}
    (OUTPUT.parent / "results_round4_summary.dat").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "summarize"))
    args = parser.parse_args()
    select() if args.mode == "select" else summarize()


if __name__ == "__main__":
    main()
''',
    "round4_assets": r'''import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import Cfg, TrueField

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

LABELS = {
    "gp_tuned": "GP sampler, selected",
    "geometry": "Midpoint tracking",
    "graph_gp": "GP-map lattice search",
    "known_tuned": "Known-field sampler",
    "graph_known": "Known-field lattice search",
    "oracle_lattice_exact": "Three-slot hindsight lattice",
    "gp_nominal": "GP sampler, nominal",
    "corridor_tuned": "Corridor reward, selected",
    "parametric_tuned": "Parametric jammer, selected",
    "reactive": "Reactive return",
    "twophase_tuned": "Two-stage schedule",
}


def interval(record):
    return f"{record['mean']:.3f} [{record['ci'][0]:.3f},{record['ci'][1]:.3f}]"


def main():
    data = json.loads((ROOT / "results_round4_summary.dat").read_text())
    methods = data["methods"]
    paired = data["paired"]
    order = tuple(LABELS)
    lines = [
        r"\begin{table*}[t]",
        r"\refstepcounter{table}\label{tab:round4main}",
        r"\noindent{\color{accessblue}\tablecapheadfont TABLE \thetable.}\ {\tablecapfont Independent 200-scenario evaluation at 3.0 Mbit/s}\par\medskip",
        r"\centering\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Method & Outage [95\% CI] & Longest [s] & Rate P05 [Mbit/s] & Energy [kJ] & Solve [ms] \\",
        r"\midrule",
    ]
    for name in order:
        m = methods[name]
        lines.append(f"{LABELS[name]} & {interval(m['outage_rate'])} & "
                     f"{m['longest_outage_s']['mean']:.1f} & "
                     f"{m['rate_p05']['mean']/1e6:.2f} & "
                     f"{m['energy_kj']['mean']:.1f} & "
                     f"{m['solve_ms_med']['mean']:.2f} " + r"\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Outage intervals use scenarios as independent units. Other entries are scenario means. The three-slot hindsight lattice uses the true field and future user positions; it is exact only within its short discrete action set.\end{flushleft}",
        r"\end{table*}",
    ])
    (PAPER / "table_round4_main.tex").write_text("\n".join(lines)+"\n")
    comparisons = ("geometry", "graph_gp", "reactive", "corridor_tuned",
                   "parametric_tuned")
    lines = [
        r"\begin{table}[t]",
        r"\caption{Selected GP sampler minus online comparator, 200 holdout scenarios}",
        r"\label{tab:round4paired}",
        r"\centering\small",
        r"\begin{tabular}{@{}lrr@{}}\toprule",
        r"Comparator & Difference [95\% CI] & $p_{\rm Holm}$ \\",
        r"\midrule",
    ]
    for name in comparisons:
        comparison = paired[f"gp_minus_{name}"]
        p = comparison["p_holm"]
        ptext = "$<0.001$" if p < .001 else f"{p:.3f}"
        lines.append(f"{LABELS[name]} & {interval(comparison)} & {ptext} " + r"\\")
    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\begin{flushleft}\footnotesize Negative differences favor the GP sampler. Holm correction covers these five planned online comparisons.\end{flushleft}",
        r"\end{table}",
    ])
    (PAPER / "table_round4_pairwise.tex").write_text("\n".join(lines)+"\n")
    stress = data["high_load"]["outage"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Locked controllers at the 4.5-Mbit/s high-load target}",
        r"\label{tab:round4stress}",
        r"\centering\small",
        r"\begin{tabular}{@{}lr@{}}\toprule",
        r"Method & Outage [95\% CI] \\",
        r"\midrule",
    ]
    for name in ("gp_tuned", "geometry", "graph_gp", "known_tuned"):
        lines.append(f"{LABELS[name]} & {interval(stress[name])} " + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    (PAPER / "table_round4_stress.tex").write_text("\n".join(lines)+"\n")
    selected = data["selected"]
    macro = [f"\\newcommand{{\\SelectedTemperature}}{{{selected['temperature']:g}}}",
             f"\\newcommand{{\\SelectedSpread}}{{{selected['sigma_v']:g}}}"]
    (PAPER / "result_numbers_round4.tex").write_text("\n".join(macro)+"\n")
    shown = ("gp_tuned", "geometry", "graph_gp", "known_tuned",
             "graph_known", "oracle_lattice_exact", "gp_nominal",
             "parametric_tuned", "corridor_tuned")
    means = np.array([methods[name]["outage_rate"]["mean"] for name in shown])
    lower = np.array([methods[name]["outage_rate"]["ci"][0] for name in shown])
    upper = np.array([methods[name]["outage_rate"]["ci"][1] for name in shown])
    colors = ["#176B5B" if name == "gp_tuned" else
              "#9B6824" if name.startswith("graph") else
              "#64748B" if name in ("known_tuned", "oracle_lattice_exact") else
              "#B8C3CD" for name in shown]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    y = np.arange(len(shown))
    ax.barh(y, means, color=colors, height=.72)
    ax.errorbar(means, y, xerr=np.vstack((means-lower, upper-means)),
                fmt="none", ecolor="black", capsize=2, lw=.8)
    ax.set_yticks(y, [LABELS[name] for name in shown], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean outage fraction and 95% scenario-level CI")
    ax.set_xlim(0, max(upper)*1.12)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(PAPER / "figures/fig11_round4_holdout.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for name, color in (("gp_tuned", "#176B5B"),
                        ("gp_nominal", "#B0522B"),
                        ("known_tuned", "#64748B")):
        values = []
        for seed in range(4000, 4200):
            path = ROOT / "results_round4/confirm" / name / f"{seed}.series.dat"
            with np.load(path) as series:
                values.append(series["trace"][:, 0].copy())
        values = np.sort(np.concatenate(values))
        ax.plot(values, np.arange(1, len(values)+1)/len(values),
                label=LABELS[name], color=color)
    ax.set_xlabel(r"all-slot $N_{\rm eff}/K_s$")
    ax.set_ylabel("cumulative fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER / "figures/fig12_round4_ess.pdf")
    plt.close(fig)
    # 胜出场景中的中位差案例只用于展示路径，不进入统计检验。
    seed = 4123
    paths = {}
    for name in ("gp_tuned", "geometry", "graph_gp"):
        path = ROOT / "results_round4/confirm" / name / f"{seed}.series.dat"
        with np.load(path) as series:
            paths[name] = {key: series[key].copy() for key in
                           ("trajectory", "user", "outages")}
    field = TrueField(np.random.default_rng(seed), Cfg)
    grid = np.linspace(0, Cfg.L, 101)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack((xx.ravel(), yy.ravel(),
                              np.full(xx.size, Cfg.H_UAV)))
    power = field(points).reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax = axes[0]
    img = ax.pcolormesh(xx, yy, power, shading="auto", cmap="YlOrRd", alpha=.65)
    colors = {"gp_tuned": "#176B5B", "geometry": "#2563A6",
              "graph_gp": "#9B6824"}
    for name, color in colors.items():
        q = paths[name]["trajectory"]
        ax.plot(q[:, 0], q[:, 1], color=color, lw=1.6,
                label=LABELS[name])
    user = paths["gp_tuned"]["user"][:Cfg.T_STEPS]
    ax.plot(user[:, 0], user[:, 1], color="#333333", ls="--",
            lw=.8, alpha=.7, label="ground user")
    ax.plot(Cfg.BS[0], Cfg.BS[1], "ks", ms=5)
    ax.plot(field.p_J[0], field.p_J[1], "kx", ms=6)
    ax.set(xlim=(0, Cfg.L), ylim=(0, Cfg.L), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(img, ax=ax, fraction=.046, pad=.03, label="true UAV-height interference [dBm]")
    ax = axes[1]
    for name, color in colors.items():
        outages = paths[name]["outages"].astype(int)
        ax.plot(np.arange(1, len(outages)+1), np.cumsum(outages),
                color=color, lw=1.7, label=LABELS[name])
    ax.set(xlim=(0, Cfg.T_STEPS), xlabel="slot", ylabel="cumulative outage slots")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(PAPER / "figures/fig13_round4_routes.pdf")
    plt.close(fig)
    path_stats = {}
    for name in ("gp_tuned", "graph_gp", "geometry", "known_tuned",
                 "gp_nominal", "corridor_tuned"):
        offsets, far, lengths = [], [], []
        for seed in range(4000, 4200):
            path = ROOT / "results_round4/confirm" / name / f"{seed}.series.dat"
            with np.load(path) as series:
                q = series["trajectory"].copy()
                user = series["user"][:len(q)].copy()
            midpoint = .5 * (Cfg.BS[:2] + user)
            distance = np.linalg.norm(q - midpoint, axis=1)
            offsets.append(float(np.mean(distance)))
            far.append(float(np.mean(distance > 500)))
            lengths.append(float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum() / 1000))
        path_stats[name] = {"mean_midpoint_offset_m": float(np.mean(offsets)),
                            "slot_fraction_offset_over_500m": float(np.mean(far)),
                            "path_length_km": float(np.mean(lengths))}
    (ROOT / "results_round4_route_summary.dat").write_text(json.dumps(path_stats, indent=2))
    lines = [r"\begin{table}[t]",
             r"\caption{Holdout route geometry at 3.0 Mbit/s}",
             r"\label{tab:round4paths}",
             r"\centering\small",
             r"\begin{tabular}{@{}lrrr@{}}\toprule",
             r"Method & Offset [m] & $>500$ m [\%] & Path [km] \\",
             r"\midrule"]
    for name in ("gp_tuned", "graph_gp", "geometry", "known_tuned",
                 "gp_nominal", "corridor_tuned"):
        row = path_stats[name]
        lines.append(f"{LABELS[name]} & {row['mean_midpoint_offset_m']:.0f} & "
                     f"{100*row['slot_fraction_offset_over_500m']:.1f} & "
                     f"{row['path_length_km']:.2f} " + r"\\")
    lines.extend([r"\bottomrule\end{tabular}",
                  r"\begin{flushleft}\footnotesize Offset is distance from the current BS--user midpoint. Entries average each scene's statistic over 200 scenes.\end{flushleft}",
                  r"\end{table}"])
    (PAPER / "table_round4_paths.tex").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
''',
    "round4_seed_audit": r'''import json
from pathlib import Path

import numpy as np
from scipy import stats

from sim import Cfg, TrueField

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = .15 * Cfg.L, .85 * Cfg.L


def first_draw(seed):
    return np.random.default_rng(seed).uniform(LOW, HIGH, size=2)


def main():
    for seed in (1000, 1029, 2000, 2199, 3000, 4099):
        actual = TrueField(np.random.default_rng(seed), Cfg).p_J
        np.testing.assert_array_equal(actual, first_draw(seed))
    x = np.array([first_draw(seed)[0] for seed in range(1000, 11000)])
    old = np.array([first_draw(seed)[0] for seed in range(1000, 1030)])
    primary = np.array([first_draw(seed)[0] for seed in range(2000, 2200)])
    new_dev = np.array([first_draw(seed)[0] for seed in range(3000, 3100)])
    new_hold = np.array([first_draw(seed)[0] for seed in range(4000, 4200)])
    rng = np.random.default_rng(935701)
    total, extreme = 0, 0
    for _ in range(100):
        means = rng.uniform(LOW, HIGH, size=(2000, 30)).mean(axis=1)
        total += len(means)
        extreme += int(np.sum(np.abs(means-1000.0) >= abs(old.mean()-1000.0)))
    result = {
        "jammer_x_range_m": [LOW, HIGH],
        "original_development_mean_m": float(old.mean()),
        "original_primary_mean_m": float(primary.mean()),
        "new_development_mean_m": float(new_dev.mean()),
        "new_holdout_mean_m": float(new_hold.mean()),
        "original_development_uniform_two_sided_monte_carlo_p": extreme / total,
        "ten_thousand_seed_uniform_ks_p": float(stats.kstest((x-LOW)/(HIGH-LOW), "uniform").pvalue),
        "ten_thousand_seed_lag_one_correlation": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
        "new_development_vs_holdout_x_ks_p": float(stats.ks_2samp(new_dev, new_hold).pvalue),
        "first_draw_replay": "exact for six audited seeds",
    }
    (ROOT / "results_round4_seed_audit.dat").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
''',
}


def _source_hashes():
    names = ("sim.py", "planner.py", "baselines.py", "experiment.py", "campaign.py")
    return {name: hashlib.sha256(MODULE_SOURCE[name[:-3]].encode("utf-8")).hexdigest()
            for name in names}


def _digest():
    campaign = sys.modules["round4_campaign"]
    names = ("round4_campaign", "graph_search_baseline", "parametric_baseline",
             "round3_campaign", "round2_mechanism")
    payload = {"core": _source_hashes(),
               "new": {name + ".py": hashlib.sha256(MODULE_SOURCE[name].encode("utf-8")).hexdigest()
                       for name in names},
               "rate": campaign.RATE, "grid": campaign.GRID,
               "extra_grid": campaign.EXTRA_GRID, "graph_grid": campaign.GRAPH_GRID}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _install_modules():
    for name, source in MODULE_SOURCE.items():
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / "src" / (name + ".py"))
        sys.modules[name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        if name == "campaign":
            module.source_hashes = _source_hashes
        if name == "round4_campaign":
            module.digest = _digest


def _run_module(name, args):
    saved = sys.argv
    try:
        sys.argv = [name + ".py", *args]
        sys.modules[name].main()
    finally:
        sys.argv = saved


def _review_worker(job):
    name, seed = job
    if "experiment" not in sys.modules:
        _install_modules()
    import experiment
    import graph_search_baseline
    from planner import GaussianField
    from round3_campaign import DiagnosticPlanner
    from sim import Cfg

    class StudyCfg(Cfg):
        R_MIN = 3e6

    class StudyPlanner(DiagnosticPlanner):
        selected_temperature = 0.05

        def __init__(self, *args, **kwargs):
            kwargs["sigma_v"] = 8.0
            super().__init__(*args, **kwargs)

    class StudyMap(GaussianField):
        def __init__(self, *args, **kwargs):
            if name == "prior_mean":
                kwargs["mean_dbm"] = -100.0
            if name == "prior_std":
                kwargs["sigma_f"] = 8.0
            super().__init__(*args, **kwargs)

        def _k(self, a, b):
            if name == "squared_exp":
                from scipy.spatial.distance import cdist
                import numpy as np
                distance = cdist(a, b)
                return self.sf ** 2 * np.exp(-0.5 * (distance / self.ell) ** 2)
            return super()._k(a, b)

    class StudyLattice(graph_search_baseline.LatticePlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, beam_width=32, depth=8, **kwargs)

    original_planner = experiment.MPPIPlanner
    original_map = experiment.GaussianField
    original_risk = graph_search_baseline.outage_prob
    try:
        study = experiment
        if name == "always_user_reports":
            study = types.ModuleType("review_experiment")
            source = MODULE_SOURCE["experiment"].replace(
                "if use_feedback and last_ok:", "if use_feedback:")
            exec(compile(source, "review_experiment.py", "exec"), study.__dict__)
        if name == "graph_with_uncertainty":
            def same_risk(*args, **kwargs):
                kwargs["use_chance"] = True
                return original_risk(*args, **kwargs)
            graph_search_baseline.outage_prob = same_risk
            study.MPPIPlanner = StudyLattice
        else:
            study.MPPIPlanner = StudyPlanner
            study.GaussianField = StudyMap
        options = {"use_feedback": name != "no_user_reports"}
        if name == "short_scale":
            options["ell_gp"] = 150.0
        elif name == "long_scale":
            options["ell_gp"] = 350.0
        elif name == "short_buffer":
            options["max_obs"] = 50
        result = study.run_episode(seed, "ours", lam=0.0, cfg=StudyCfg,
                                   grid_reference=11, **options)
    finally:
        experiment.MPPIPlanner = original_planner
        experiment.GaussianField = original_map
        graph_search_baseline.outage_prob = original_risk
    return {"condition": name, "seed": seed,
            "outage_rate": result["outage_rate"],
            "ahead_rmse": result["ahead_rmse"],
            "solve_ms_med": result["solve_ms_med"]}


def _review_checks(workers):
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    from scipy import stats
    conditions = ("no_user_reports", "always_user_reports",
                  "short_scale", "long_scale",
                  "short_buffer", "prior_mean", "prior_std",
                  "squared_exp", "graph_with_uncertainty")
    seeds = range(3000, 3030)
    target = ROOT / "results_review"
    target.mkdir(exist_ok=True)
    jobs = [(name, seed) for name in conditions for seed in seeds
            if not (target / name / f"{seed}.dat").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, record in enumerate(pool.map(_review_worker, jobs), 1):
            path = target / record["condition"] / f"{record['seed']}.dat"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(record, indent=2))
            if index % 20 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} reviewer runs complete", flush=True)
    baseline = np.array([
        json.loads((ROOT / "results_round4/calibrate/cal_tau0.05_sigma8" /
                    f"{seed}.metrics.dat").read_text())["metrics"]["outage_rate"]
        for seed in seeds])
    baseline_rmse = np.array([
        json.loads((ROOT / "results_round4/calibrate/cal_tau0.05_sigma8" /
                    f"{seed}.metrics.dat").read_text())["metrics"]["ahead_rmse"]
        for seed in seeds])
    summary = {"seeds": [3000, 3029],
               "baseline_mean_outage": float(np.mean(baseline)),
               "baseline_mean_ahead_rmse": float(np.mean(baseline_rmse)),
               "conditions": {}}
    for name in conditions:
        rows = [json.loads((target / name / f"{seed}.dat").read_text())
                for seed in seeds]
        values = np.array([r["outage_rate"] for r in rows])
        difference = values - baseline
        margin = stats.t.ppf(0.975, len(seeds)-1) * stats.sem(difference)
        summary["conditions"][name] = {
            "mean_outage": float(np.mean(values)),
            "paired_delta": float(np.mean(difference)),
            "paired_ci": [float(np.mean(difference)-margin),
                          float(np.mean(difference)+margin)],
            "mean_ahead_rmse": float(np.nanmean([r["ahead_rmse"] for r in rows])),
        }
    fixed_graph = np.array([
        json.loads((ROOT / "results_round4/graph_calibrate/graph_b32_h8" /
                    f"{seed}.metrics.dat").read_text())["metrics"]["outage_rate"]
        for seed in seeds])
    changed_graph = np.array([
        json.loads((target / "graph_with_uncertainty" /
                    f"{seed}.dat").read_text())["outage_rate"]
        for seed in seeds])
    delta = changed_graph - fixed_graph
    margin = stats.t.ppf(0.975, len(seeds)-1) * stats.sem(delta)
    summary["graph_fixed_spread_mean_outage"] = float(fixed_graph.mean())
    summary["graph_uncertainty_minus_fixed"] = {
        "mean": float(delta.mean()),
        "ci": [float(delta.mean()-margin), float(delta.mean()+margin)]}
    (ROOT / "results_review_summary.dat").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _feedback_holdout(workers):
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    from scipy import stats
    seeds = range(4000, 4200)
    conditions = ("no_user_reports", "always_user_reports")
    target = ROOT / "results_review" / "feedback_holdout"
    target.mkdir(parents=True, exist_ok=True)
    jobs = [(condition, seed) for condition in conditions for seed in seeds
            if not (target / f"{condition}_{seed}.dat").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, record in enumerate(pool.map(_review_worker, jobs), 1):
            (target / f"{record['condition']}_{record['seed']}.dat").write_text(
                json.dumps(record, indent=2))
            if index % 20 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} feedback runs complete", flush=True)
    baseline = np.array([
        json.loads((ROOT / "results_round4/confirm/gp_tuned" /
                    f"{seed}.metrics.dat").read_text())["metrics"]["outage_rate"]
        for seed in seeds])
    baseline_rmse = np.array([
        json.loads((ROOT / "results_round4/confirm/gp_tuned" /
                    f"{seed}.metrics.dat").read_text())["metrics"]["ahead_rmse"]
        for seed in seeds])
    result = {"seeds": [seeds.start, seeds.stop - 1],
              "baseline_mean_outage": float(baseline.mean()),
              "baseline_mean_ahead_rmse": float(baseline_rmse.mean()),
              "conditions": {}}
    for condition in conditions:
        rows = [json.loads((target / f"{condition}_{seed}.dat").read_text())
                for seed in seeds]
        outage = np.array([r["outage_rate"] for r in rows])
        difference = outage - baseline
        margin = stats.t.ppf(.975, len(difference)-1) * stats.sem(difference)
        result["conditions"][condition] = {
            "mean_outage": float(outage.mean()),
            "paired_difference": float(difference.mean()),
            "paired_ci": [float(difference.mean()-margin),
                          float(difference.mean()+margin)],
            "mean_ahead_rmse": float(np.mean([r["ahead_rmse"] for r in rows]))}
    (ROOT / "results_review_feedback_holdout.dat").write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def _risk_audit_worker(seed):
    if "planner" not in sys.modules:
        _install_modules()
    import numpy as np
    from planner import GaussianField, outage_prob
    from sim import Cfg, TrueField
    class StudyCfg(Cfg):
        R_MIN = 3e6
    path = ROOT / "results_round4/confirm/gp_tuned" / f"{seed}.series.dat"
    with np.load(path) as series:
        route = series["trajectory"].copy()
        user = series["user"].copy()
        observed = series["outages"].astype(int).copy()
    field = TrueField(np.random.default_rng(seed), StudyCfg)
    errors = np.random.default_rng(np.random.SeedSequence([seed, 173])).normal(
        0, StudyCfg.SIGMA_Y, (StudyCfg.T_STEPS, 2))
    gp = GaussianField(ell=250.0, sigma_y=StudyCfg.SIGMA_Y, max_obs=150)
    predictions = []
    for t, (q, g) in enumerate(zip(route, user)):
        q3 = np.r_[q, StudyCfg.H_UAV]
        g3 = np.r_[g, 0.0]
        mu_q, sd_q = gp.predict(q3[None, :])
        mu_g, sd_g = gp.predict(g3[None, :])
        score = outage_prob(q, g, mu_q, sd_q, mu_g, sd_g, StudyCfg)
        predictions.append(float(score[0]))
        gp.add(q3, float(field(q3)[0]) + errors[t, 0])
        if not observed[t]:
            gp.add(g3, float(field(g3)[0]) + errors[t, 1])
    return seed, predictions, observed.tolist()


def _risk_audit(workers):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    target = ROOT / "results_review/risk_audit"
    target.mkdir(parents=True, exist_ok=True)
    seeds = range(4000, 4200)
    jobs = [seed for seed in seeds if not (target / f"{seed}.dat").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (seed, predicted, observed) in enumerate(
                pool.map(_risk_audit_worker, jobs), 1):
            (target / f"{seed}.dat").write_text(json.dumps(
                {"seed": seed, "predicted": predicted, "outage": observed}))
            if index % 20 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} risk-audit scenes complete",
                      flush=True)
    rows = [json.loads((target / f"{seed}.dat").read_text()) for seed in seeds]
    p = np.concatenate([r["predicted"] for r in rows])
    y = np.concatenate([r["outage"] for r in rows])
    edges = np.linspace(0, 1, 11)
    bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, 9)
    calibration = []
    for j in range(10):
        mask = bins == j
        calibration.append({"range": [float(edges[j]), float(edges[j+1])],
                            "n": int(mask.sum()),
                            "mean_score": float(p[mask].mean()) if mask.any() else None,
                            "observed_outage": float(y[mask].mean()) if mask.any() else None})
    actual = float(y.mean())
    result = {"seeds": [4000, 4199], "slots": int(len(p)),
              "mean_score": float(p.mean()), "observed_outage": actual,
              "brier_score": float(np.mean((p-y)**2)),
              "constant_rate_brier": float(np.mean((actual-y)**2)),
              "expected_calibration_error": float(sum(
                  row["n"]/len(p)*abs(row["mean_score"]-row["observed_outage"])
                  for row in calibration if row["n"])),
              "bins": calibration}
    (ROOT / "results_review_risk_audit.dat").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def _boundary_audit():
    import numpy as np
    from scipy import stats
    from sim import Cfg
    root = ROOT / "results_round4/confirm"
    output = {}
    clean_gp_seeds = []
    for method in sorted(p.name for p in root.iterdir() if p.is_dir()):
        violations = 0
        edge_violations = 0
        transitions = 0
        max_change = 0.0
        for path in (root / method).glob("*.series.dat"):
            with np.load(path) as series:
                q = series["trajectory"]
            velocity = np.diff(q, axis=0) / Cfg.DT
            changes = np.linalg.norm(np.diff(velocity, axis=0), axis=1)
            bad = changes > Cfg.A_MAX * Cfg.DT + 1e-8
            edge = np.any((q[1:-1] <= 1e-8) | (q[1:-1] >= Cfg.L-1e-8),
                          axis=1) | np.any(
                (q[2:] <= 1e-8) | (q[2:] >= Cfg.L-1e-8), axis=1)
            violations += int(bad.sum())
            edge_violations += int((bad & edge).sum())
            transitions += len(changes)
            max_change = max(max_change, float(changes.max()))
            if method == "gp_tuned" and not bad.any():
                clean_gp_seeds.append(int(path.name.split(".")[0]))
        output[method] = {"violations": violations,
                          "edge_related": edge_violations,
                          "transitions": transitions,
                          "maximum_velocity_change_mps": max_change}
    clean_gp_seeds.sort()
    def outage(method, seed):
        path = root / method / f"{seed}.metrics.dat"
        return json.loads(path.read_text())["metrics"]["outage_rate"]
    paired = np.array([outage("gp_tuned", seed) - outage("geometry", seed)
                       for seed in clean_gp_seeds])
    margin = stats.t.ppf(.975, len(paired)-1) * stats.sem(paired)
    output["gp_pair_without_boundary_exceedance"] = {
        "scenes": len(clean_gp_seeds), "paired_mean": float(paired.mean()),
        "paired_ci": [float(paired.mean()-margin), float(paired.mean()+margin)]}
    (ROOT / "results_review_boundary_audit.dat").write_text(
        json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


def _safe_velocity(position, previous, proposed, cfg):
    import numpy as np
    position = np.asarray(position, float)
    previous = np.asarray(previous, float)
    velocity = np.asarray(proposed, float).copy()
    limit = cfg.A_MAX * cfg.DT
    axis_deceleration = cfg.A_MAX / np.sqrt(2.0)
    left = np.maximum(position, 0.0)
    right = np.maximum(cfg.L - position, 0.0)
    low = -np.maximum(np.sqrt((axis_deceleration * cfg.DT) ** 2
                               + 2.0 * axis_deceleration * left)
                      - axis_deceleration * cfg.DT, 0.0)
    high = np.maximum(np.sqrt((axis_deceleration * cfg.DT) ** 2
                              + 2.0 * axis_deceleration * right)
                      - axis_deceleration * cfg.DT, 0.0)
    low = np.maximum(low, -left / cfg.DT)
    high = np.minimum(high, right / cfg.DT)
    shape = velocity.shape[:-1]
    low = np.broadcast_to(low, velocity.shape)
    high = np.broadcast_to(high, velocity.shape)
    previous = np.broadcast_to(previous, velocity.shape)
    anchor = np.clip(previous, low, high)
    if np.any(np.linalg.norm(anchor - previous, axis=-1) > limit + 1e-7):
        raise RuntimeError("no feasible boundary-safe acceleration at this state")
    velocity = np.clip(velocity, low, high)
    for _ in range(30):
        delta = velocity - previous
        length = np.linalg.norm(delta, axis=-1, keepdims=True)
        velocity = previous + delta * np.minimum(1.0, limit / np.maximum(length, 1e-12))
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= np.minimum(1.0, cfg.V_MAX / np.maximum(speed, 1e-12))
        velocity = np.clip(velocity, low, high)
    acceleration = np.linalg.norm(velocity - previous, axis=-1)
    speed = np.linalg.norm(velocity, axis=-1)
    if (np.any(acceleration > limit + 1e-6) or
            np.any(speed > cfg.V_MAX + 1e-6) or
            np.any(position + velocity * cfg.DT < -1e-6) or
            np.any(position + velocity * cfg.DT > cfg.L + 1e-6)):
        raise RuntimeError("boundary-safe projection failed")
    return velocity.reshape((*shape, 2))


def _safe_rollout(q, candidate, previous, cfg):
    import numpy as np
    positions = np.broadcast_to(q, (len(candidate), 2)).copy()
    velocities = np.broadcast_to(previous, positions.shape).copy()
    route = np.empty_like(candidate)
    for h in range(candidate.shape[1]):
        velocities = _safe_velocity(positions, velocities, candidate[:, h], cfg)
        positions = positions + velocities * cfg.DT
        route[:, h] = positions
        candidate[:, h] = velocities
    return route


def _boundary_safe_worker(job):
    method, seed = job
    if "experiment" not in sys.modules:
        _install_modules()
    import numpy as np
    from round3_campaign import DiagnosticPlanner
    from sim import Cfg

    class StudyCfg(Cfg):
        R_MIN = 3e6

    planner_source = MODULE_SOURCE["planner"]
    old = "Q = q[None, None, :] + np.cumsum(U * cfg.DT, axis=1)\n        Q = np.clip(Q, 0.0, cfg.L)"
    new = "Q = safe_rollout(q, U, self.last_velocity, cfg)"
    if planner_source.count(old) != 1:
        raise RuntimeError("planner rollout source changed")
    safe_planner_module = types.ModuleType("safe_planner")
    safe_planner_module.safe_rollout = _safe_rollout
    exec(compile(planner_source.replace(old, new), "safe_planner.py", "exec"),
         safe_planner_module.__dict__)

    class SafePlanner(safe_planner_module.MPPIPlanner):
        def __init__(self, *args, **kwargs):
            kwargs["temperature"] = 0.05
            kwargs["sigma_v"] = 8.0
            super().__init__(*args, **kwargs)

    experiment_source = MODULE_SOURCE["experiment"]
    old = "q_next = np.clip(q + v * cfg.DT, 0.0, cfg.L)"
    new = "v = safe_velocity(q, v_prev, v, cfg)\n        q_next = q + v * cfg.DT"
    if experiment_source.count(old) != 1:
        raise RuntimeError("experiment motion source changed")
    safe_experiment = types.ModuleType("safe_experiment")
    safe_experiment.safe_velocity = _safe_velocity
    exec(compile(experiment_source.replace(old, new), "safe_experiment.py", "exec"),
         safe_experiment.__dict__)
    safe_experiment.MPPIPlanner = SafePlanner
    result = safe_experiment.run_episode(seed, method, lam=0.0, cfg=StudyCfg,
                                         grid_reference=11, record=True)
    initial = 0.5 * (StudyCfg.BS[:2] + result["_gtraj"][0])
    path = np.vstack((initial, result["_traj"]))
    velocity = np.diff(path, axis=0) / StudyCfg.DT
    changes = np.linalg.norm(np.diff(np.vstack((np.zeros(2), velocity)), axis=0), axis=1)
    return {"method": method, "seed": seed,
            "outage_rate": float(result["outage_rate"]),
            "max_realized_velocity_change": float(changes.max()),
            "acceleration_violations": int(np.sum(changes > StudyCfg.A_MAX * StudyCfg.DT + 1e-6)),
            "maximum_speed": float(np.linalg.norm(velocity, axis=1).max())}


def _boundary_safe_checks(workers, holdout=False):
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    from scipy import stats
    methods = ("ours",) if holdout else ("ours", "oblivious")
    seeds = range(4000, 4200) if holdout else range(3000, 3030)
    label = "boundary_safe_holdout" if holdout else "boundary_safe"
    target = ROOT / "results_review" / label
    target.mkdir(parents=True, exist_ok=True)
    jobs = [(method, seed) for method in methods for seed in seeds
            if not (target / f"{method}_{seed}.dat").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(_boundary_safe_worker, jobs), 1):
            (target / f"{row['method']}_{row['seed']}.dat").write_text(
                json.dumps(row, indent=2))
            if index % 10 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} boundary-safe runs complete", flush=True)
    rows = {method: [json.loads((target / f"{method}_{seed}.dat").read_text())
                     for seed in seeds] for method in methods}
    if holdout:
        rows["oblivious"] = [
            {"outage_rate": json.loads((ROOT / "results_round4/confirm/geometry" /
                                       f"{seed}.metrics.dat").read_text())["metrics"]["outage_rate"],
             "acceleration_violations": 0}
            for seed in seeds]
    a = np.array([r["outage_rate"] for r in rows["ours"]])
    b = np.array([r["outage_rate"] for r in rows["oblivious"]])
    delta = a - b
    margin = stats.t.ppf(.975, len(delta)-1) * stats.sem(delta)
    summary = {"seeds": [seeds.start, seeds.stop - 1],
               "midpoint_source": "original holdout, zero acceleration violations" if holdout else "boundary-safe rerun",
               "gp_mean_outage": float(a.mean()),
               "midpoint_mean_outage": float(b.mean()),
               "paired_difference": float(delta.mean()),
               "paired_ci": [float(delta.mean()-margin), float(delta.mean()+margin)],
               "acceleration_violations": int(sum(r["acceleration_violations"]
                                                   for method in methods for r in rows[method])),
               "maximum_realized_velocity_change": float(max(
                   r["max_realized_velocity_change"] for method in methods for r in rows[method]))}
    (ROOT / f"results_review_{label}.dat").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _robustness_worker(job):
    condition, method, seed = job
    if "experiment" not in sys.modules:
        _install_modules()
    import experiment
    from round3_campaign import DiagnosticPlanner
    from sim import Cfg
    changes = {
        "jammer_exponent_2p5": {"ALPHA_J": 2.5},
        "shadow_std_9": {"SHADOW_STD": 9.0},
        "no_rectangles": {"N_BLOCK": 0},
        "altitude_120": {"H_UAV": 120.0},
        "user_speed_7": {"V_G": 7.0},
    }
    cfg = type("StudyCfg", (Cfg,), {"R_MIN": 3e6, **changes[condition]})
    class StudyPlanner(DiagnosticPlanner):
        selected_temperature = 0.05
        def __init__(self, *args, **kwargs):
            kwargs["sigma_v"] = 8.0
            super().__init__(*args, **kwargs)
    old = experiment.MPPIPlanner
    experiment.MPPIPlanner = StudyPlanner
    try:
        result = experiment.run_episode(
            seed, "ours" if method == "gp" else "oblivious",
            cfg=cfg, grid_reference=11)
    finally:
        experiment.MPPIPlanner = old
    return {"condition": condition, "method": method, "seed": seed,
            "outage_rate": result["outage_rate"]}


def _robustness_checks(workers):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats
    conditions = ("jammer_exponent_2p5", "shadow_std_9", "no_rectangles",
                  "altitude_120", "user_speed_7")
    seeds = range(3000, 3030)
    target = ROOT / "results_review/robustness"
    target.mkdir(parents=True, exist_ok=True)
    jobs = [(condition, method, seed)
            for condition in conditions for method in ("gp", "geometry")
            for seed in seeds
            if not (target / condition / method / f"{seed}.dat").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(_robustness_worker, jobs), 1):
            path = target / row["condition"] / row["method"] / f"{row['seed']}.dat"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, indent=2))
            if index % 20 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} robustness runs complete",
                      flush=True)
    summary = {"seeds": [3000, 3029], "conditions": {}}
    for condition in conditions:
        values = {}
        for method in ("gp", "geometry"):
            values[method] = np.array([
                json.loads((target / condition / method /
                            f"{seed}.dat").read_text())["outage_rate"]
                for seed in seeds])
        difference = values["gp"] - values["geometry"]
        margin = stats.t.ppf(.975, len(seeds)-1) * stats.sem(difference)
        summary["conditions"][condition] = {
            "gp_mean": float(values["gp"].mean()),
            "geometry_mean": float(values["geometry"].mean()),
            "paired_gp_minus_geometry": float(difference.mean()),
            "paired_ci": [float(difference.mean()-margin),
                          float(difference.mean()+margin)]}
    (ROOT / "results_review_robustness.dat").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _review_figures():
    import matplotlib.pyplot as plt
    import numpy as np
    from sim import Cfg, TrueField
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    figures = ROOT / "paper/figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.set(xlim=(0, 10), ylim=(0, 7.5))
    ax.axis("off")
    nodes = ((1.1, 3.6, "BS", "#2563A6"),
             (5.0, 3.6, "UAV relay", "#176B5B"),
             (8.9, 3.6, "Moving user", "#9B6824"),
             (5.0, 6.5, "Online map", "#53627A"),
             (5.0, 0.6, "Jammer", "#9B2E42"))
    for x, y, label, color in nodes:
        ax.add_patch(FancyBboxPatch((x-0.85, y-0.3), 1.7, 0.6,
                     boxstyle="round,pad=0.06", linewidth=1.2,
                     edgecolor=color, facecolor="white", zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                color=color, zorder=4)
    def arrow(start, end, color="#333333", style="-", curve=0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                     connectionstyle=f"arc3,rad={curve}", mutation_scale=9,
                     linewidth=1.2, linestyle=style, color=color, zorder=2))
    arrow((1.95, 3.6), (4.15, 3.6))
    arrow((5.85, 3.6), (8.05, 3.6))
    ax.text(3.05, 3.88, "BS–UAV hop", ha="center", fontsize=7.5)
    ax.text(6.95, 3.88, "UAV–user hop", ha="center", fontsize=7.5)
    arrow((5.0, 3.92), (5.0, 6.18), "#176B5B")
    ax.text(3.48, 5.2, "UAV radio reading", ha="center", fontsize=7.1)
    arrow((8.75, 3.92), (5.85, 6.5), "#6B7280", ":")
    ax.text(8.05, 5.75, "User report after service", ha="center", fontsize=7.1)
    arrow((8.05, 3.27), (5.85, 3.27), "#2563A6", ":")
    ax.text(6.95, 2.72, "Independent position feed", ha="center", fontsize=7.0)
    arrow((4.8, 0.93), (4.8, 3.27), "#9B2E42", "--")
    arrow((5.55, 0.9), (8.75, 3.27), "#9B2E42", "--")
    ax.text(4.2, 1.9, "Interference", ha="right", fontsize=7.1,
            color="#9B2E42")
    ax.text(7.75, 1.38, "Interference", ha="center", fontsize=7.1,
            color="#9B2E42")
    fig.tight_layout()
    fig.savefig(figures / "fig_system_model.pdf", bbox_inches="tight")
    plt.close(fig)

    seed = 4123
    paths = {}
    for name in ("gp_tuned", "geometry", "graph_gp"):
        with np.load(ROOT / "results_round4/confirm" / name /
                     f"{seed}.series.dat") as series:
            paths[name] = {key: series[key].copy()
                           for key in ("trajectory", "user", "outages")}
    field = TrueField(np.random.default_rng(seed), Cfg)
    grid = np.linspace(0, Cfg.L, 101)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack((xx.ravel(), yy.ravel(),
                              np.full(xx.size, Cfg.H_UAV)))
    power = field(points).reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax = axes[0]
    img = ax.pcolormesh(xx, yy, power, shading="auto", cmap="YlOrRd", alpha=.65)
    colors = {"gp_tuned": "#176B5B", "geometry": "#2563A6",
              "graph_gp": "#9B6824"}
    names = {"gp_tuned": "GP sampler", "geometry": "Midpoint",
             "graph_gp": "GP lattice"}
    for name, color in colors.items():
        q = paths[name]["trajectory"]
        ax.plot(q[:, 0], q[:, 1], color=color, lw=1.6, label=names[name])
    user = paths["gp_tuned"]["user"][:Cfg.T_STEPS]
    ax.plot(user[:, 0], user[:, 1], color="#333333", ls="--",
            lw=.8, label="Ground user")
    ax.plot(Cfg.BS[0], Cfg.BS[1], "ks", ms=5, label="BS")
    ax.plot(field.p_J[0], field.p_J[1], "kx", ms=6, label="Jammer")
    ax.set(xlim=(0, Cfg.L), ylim=(0, Cfg.L), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=.85)
    fig.colorbar(img, ax=ax, fraction=.046, pad=.03,
                 label="true UAV-height interference [dBm]")
    ax = axes[1]
    for name, color in colors.items():
        outages = paths[name]["outages"].astype(int)
        ax.plot(np.arange(1, len(outages)+1), np.cumsum(outages),
                color=color, lw=1.7, label=names[name])
    ax.set(xlim=(0, Cfg.T_STEPS), xlabel="slot",
           ylabel="cumulative outage slots")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures / "fig13_round4_routes.pdf")
    plt.close(fig)
    audit_path = ROOT / "results_review_risk_audit.dat"
    if audit_path.exists():
        bins = json.loads(audit_path.read_text())["bins"]
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        centers = [row["mean_score"] for row in bins]
        observed = [row["observed_outage"] for row in bins]
        counts = [row["n"] for row in bins]
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        axes[0].plot(centers, observed, "o-", color="#176B5B",
                     label="observed")
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="mean planning score",
                    ylabel="observed outage fraction")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=.2)
        axes[1].bar(np.arange(10), counts, color="#2563A6")
        axes[1].set_yscale("log")
        axes[1].set(xlabel="score bin (0.1 width)", ylabel="decision slots (log)",
                    xticks=np.arange(0, 10, 2))
        fig.tight_layout()
        fig.savefig(figures / "fig_score_reliability.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="UAV relay experiment and paper assets")
    parser.add_argument("command", choices=(
        "calibrate", "calibrate_extra", "graph_calibrate", "select",
        "confirm", "audit", "stress", "summarize", "assets",
        "seed-audit", "verify", "review-checks", "risk-audit", "boundary-audit",
        "robustness", "boundary-safe", "boundary-safe-holdout", "feedback-holdout",
        "review-figures", "all"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    _install_modules()
    selected = ROOT / "results_round4" / "selection.dat"
    if args.command == "verify":
        record = json.loads(selected.read_text())
        if _digest() != record["signature"]:
            raise RuntimeError("source signature differs from the saved experiment")
        print("source signature verified:", _digest())
        return
    if args.command == "review-checks":
        _review_checks(args.workers)
        return
    if args.command == "feedback-holdout":
        _feedback_holdout(args.workers)
        return
    if args.command == "risk-audit":
        _risk_audit(args.workers)
        return
    if args.command == "boundary-audit":
        _boundary_audit()
        return
    if args.command == "robustness":
        _robustness_checks(args.workers)
        return
    if args.command in ("boundary-safe", "boundary-safe-holdout"):
        _boundary_safe_checks(args.workers, args.command == "boundary-safe-holdout")
        return
    if args.command == "review-figures":
        _review_figures()
        return
    if args.command == "all":
        commands = ("seed-audit", "calibrate", "calibrate_extra",
                    "graph_calibrate", "select", "confirm", "audit",
                    "stress", "summarize", "assets")
    else:
        commands = (args.command,)
    for command in commands:
        if command in ("calibrate", "calibrate_extra", "graph_calibrate",
                       "confirm", "audit", "stress"):
            _run_module("round4_campaign", [command, "--workers", str(args.workers)])
        elif command in ("select", "summarize"):
            _run_module("round4_analysis", [command])
        elif command == "assets":
            sys.modules["round4_assets"].main()
            _review_figures()
        else:
            sys.modules["round4_seed_audit"].main()


if __name__ == "__main__":
    main()

"""Implementation of the Spectral--Topographic Cross-Coupled Echo State Network."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from .electrodes import ELECTRODE_COORDS, pairwise_distances


BANDS = ("delta", "theta", "alpha", "beta", "gamma")


@dataclass
class STCSpec:
    electrodes: Sequence[str]
    cfc: Dict[Tuple[str, str, str], float]  # (electrode, b1, b2) -> coupling strength
    p_rew: float = 0.12
    tau: float = 0.01
    kappa: Dict[str, float] | float = 0.7
    sigma: Dict[str, float] | float = 5.0
    g0: float = 0.3
    lambda0: float = 0.5
    varrho: float = 0.95


class STCESN:
    def __init__(self, spec: STCSpec, seed: int | None = None):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.n_e = len(spec.electrodes)
        self.n_res = self.n_e * len(BANDS)
        self.band_indices = {
            b: slice(i * self.n_e, (i + 1) * self.n_e)
            for i, b in enumerate(BANDS)
        }
        self.W = self._build_recurrent()
        self.leak = self._build_leak_rates()
        # scale to have spectral radius < varrho
        eigvals = np.linalg.eigvals(self.W)
        rho = max(abs(eigvals))
        self.W = self.spec.varrho * self.W / rho

    def _band_param(self, param: Dict[str, float] | float, band: str) -> float:
        if isinstance(param, dict):
            return param[band]
        return float(param)

    def _build_recurrent(self) -> np.ndarray:
        n = self.n_res
        W = np.zeros((n, n))
        dists = pairwise_distances(tuple(self.spec.electrodes))
        for b in BANDS:
            idx = self.band_indices[b]
            sigma = self._band_param(self.spec.sigma, b)
            kappa = self._band_param(self.spec.kappa, b)
            block = kappa * np.exp(-(dists ** 2) / (sigma ** 2))
            noise = self.spec.tau * self.rng.standard_normal(block.shape)
            block += noise
            for i in range(self.n_e):
                for j in range(self.n_e):
                    if i == j:
                        continue
                    if self.rng.random() < self.spec.p_rew:
                        # rewire weight to a random long-range contact
                        k = self.rng.integers(0, self.n_e)
                        block[i, j] = block[i, k]
            W[idx, idx] = block
        # Cross-frequency coupling
        for e in self.spec.electrodes:
            for b1 in BANDS:
                for b2 in BANDS:
                    if b1 == b2:
                        continue
                    key = (e, b1, b2)
                    if key not in self.spec.cfc:
                        continue
                    i = self.spec.electrodes.index(e) + self.band_indices[b1].start
                    j = self.spec.electrodes.index(e) + self.band_indices[b2].start
                    W[j, i] = self.spec.cfc[key]
        return W

    def _build_leak_rates(self) -> np.ndarray:
        leak = np.full(self.n_res, self.spec.lambda0)
        # valence (frontal alpha asymmetry)
        left = {"F3", "F7", "FL"}
        right = {"F4", "F8", "FR"}
        for e in left:
            if e in self.spec.electrodes:
                idx = self.spec.electrodes.index(e) + self.band_indices["alpha"].start
                leak[idx] = self.spec.lambda0 * (1 + np.tanh(self.spec.g0))
        for e in right:
            if e in self.spec.electrodes:
                idx = self.spec.electrodes.index(e) + self.band_indices["alpha"].start
                leak[idx] = self.spec.lambda0 * (1 - np.tanh(self.spec.g0))
        return leak

    def step(self, x: np.ndarray, u: np.ndarray, Win: np.ndarray,
             Wfb: np.ndarray | None = None, y: np.ndarray | None = None,
             b: np.ndarray | None = None) -> np.ndarray:
        pre = self.W @ x + Win @ u
        if Wfb is not None and y is not None:
            pre += Wfb @ y
        if b is not None:
            pre += b
        x = (1 - self.leak) * x + self.leak * np.tanh(pre)
        return x

    def run(self, U: np.ndarray, Win: np.ndarray,
            Wfb: np.ndarray | None = None, Y_fb: np.ndarray | None = None,
            b: np.ndarray | None = None,
            x0: np.ndarray | None = None) -> np.ndarray:
        T = U.shape[0]
        x = np.zeros(self.n_res) if x0 is None else x0.copy()
        states = np.zeros((T, self.n_res))
        for t in range(T):
            y_fb = Y_fb[t] if Y_fb is not None else None
            x = self.step(x, U[t], Win, Wfb, y_fb, b)
            states[t] = x
        return states

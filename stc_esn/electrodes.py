"""Utility functions for 10-20 EEG electrode geometry."""

from typing import Dict, Tuple
import numpy as np

# 10-20 electrode positions in spherical coordinates (theta, phi) in degrees.
# This dictionary is not exhaustive but covers most electrodes in
# common emotion datasets. Coordinates approximate positions on a unit sphere.
ELECTRODE_COORDS: Dict[str, Tuple[float, float]] = {
    "Fp1": (106, -72),
    "Fp2": (74, -72),
    "F3": (124, -54),
    "F4": (56, -54),
    "C3": (144, 0),
    "C4": (36, 0),
    "P3": (124, 54),
    "P4": (56, 54),
    "O1": (106, 72),
    "O2": (74, 72),
    "F7": (150, -72),
    "F8": (30, -72),
    "T7": (180, 0),
    "T8": (0, 0),
    "P7": (150, 72),
    "P8": (30, 72),
    "Fz": (90, -54),
    "Cz": (90, 0),
    "Pz": (90, 54),
    "Oz": (90, 90),
}


def geodesic_distance(e1: str, e2: str) -> float:
    """Geodesic distance on the unit sphere between two electrodes."""
    th1, ph1 = np.radians(ELECTRODE_COORDS[e1])
    th2, ph2 = np.radians(ELECTRODE_COORDS[e2])
    v1 = np.array([
        np.sin(th1) * np.cos(ph1),
        np.sin(th1) * np.sin(ph1),
        np.cos(th1),
    ])
    v2 = np.array([
        np.sin(th2) * np.cos(ph2),
        np.sin(th2) * np.sin(ph2),
        np.cos(th2),
    ])
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def pairwise_distances(electrodes: Tuple[str, ...]) -> np.ndarray:
    """Matrix of pairwise geodesic distances between electrodes."""
    n = len(electrodes)
    dists = np.zeros((n, n))
    for i, ei in enumerate(electrodes):
        for j, ej in enumerate(electrodes):
            if j <= i:
                continue
            d = geodesic_distance(ei, ej)
            dists[i, j] = dists[j, i] = d
    return dists

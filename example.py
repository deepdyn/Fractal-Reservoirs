"""Example usage of the Spectral--Topographic Cross-Coupled ESN."""
import numpy as np
from stc_esn.reservoir import STCESN, STCSpec, BANDS
from stc_esn.electrodes import ELECTRODE_COORDS

# Define electrode list (10-20 montage subset)
electrodes = list(ELECTRODE_COORDS.keys())

# Fake cross-frequency coupling dictionary
cfc = {}
for e in electrodes:
    for b1 in BANDS:
        for b2 in BANDS:
            if b1 != b2:
                cfc[(e, b1, b2)] = 0.05

spec = STCSpec(electrodes=electrodes, cfc=cfc)
reservoir = STCESN(spec, seed=42)

# Random input: T time steps, five band envelopes per electrode
T = 10
U = np.random.randn(T, len(electrodes) * len(BANDS))
Win = np.eye(reservoir.n_res)

states = reservoir.run(U, Win)
print("Reservoir states shape:", states.shape)

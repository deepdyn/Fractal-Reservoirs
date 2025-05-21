# Fractal-Reservoirs
This repository contains exploratory implementations of reservoir
computing architectures.  The `stc_esn` package provides a reference
implementation of the **Spectral--Topographic Cross-Coupled Echo State
Network (STC-ESN)** as described in recent EEG emotion-recognition
literature.  It constructs a reservoir that mirrors the scalp layout and
cross-frequency interactions of cortical rhythms.

The example below shows how to instantiate the STC-ESN and run it on
dummy inputs:

```bash
python example.py
```

The script prints the shape of the generated reservoir state matrix.
Dependencies are kept minimal (`numpy` only).

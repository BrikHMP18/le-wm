# Simulator and data-format provenance

Prepared October 5, 2026.

## TeachingCartpole

Physics and drawing are adapted from
[dmeger/TeachingCartpole](https://github.com/dmeger/TeachingCartpole),
`cartpole_envs.py`, single-pole `CartPole` class. The assignment identifies its
vendored simulator revision as `ec8a7fa`.

The source inspected locally was
`comp-765/assignments/assignment-1/q2/TeachingCartpole/cartpole_envs.py`, SHA-256:

```text
4dea80bf0e9b11b5ff476f4bfbdf3260b4779ddbd64f0dd4df7146d25707e531
```

The upstream simulator's attribution states:

> Copyright (c) 2017, Juan Camilo Gamboa Higuera, Anqi Xu, Victor Barbaros,
> Alex Chatron-Michaud, David Meger

The teaching simulator is by David Meger; its GUI credits Wesley Fernandes.
The simulator header permits use for any purpose and welcomes source
acknowledgement. This experiment preserves that acknowledgement and changes
only the single-pole wrapper/rendering and collection plumbing: owned state
arrays, explicit solver-success checks, clean reset, no GUI, and RGB resize with
letterboxing. The dynamics, constants, state order, and native geometry follow
the assignment simulator. Friction is **b = 1.0**, matching the implementation;
the handout lists 0.1.

## LeWM and stable-worldmodel

- Fork base: LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Data schema inspected: stable-worldmodel
  `21446f1ede6d5284e981bd7b47f432b994e6d812`,
  [`data/formats/hdf5.py`](https://github.com/galilai-group/stable-worldmodel/blob/21446f1ede6d5284e981bd7b47f432b994e6d812/stable_worldmodel/data/formats/hdf5.py)
  and `data/dataset.py`.

The collector uses h5py directly to keep data collection independent of PyTorch,
CUDA, and training dependencies. Its flat columns and `ep_len`/`ep_offset`
metadata follow the native HDF5 reader. Compression is built-in lossless gzip;
the collector does not require the hdf5plugin package.

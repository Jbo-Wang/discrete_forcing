# Discrete Forcing

Code for training and evaluating a discrete-to-continuous vision-language-action
policy. The current release includes the LIBERO implementation, built on StarVLA.

## News

<!-- News will be added here. -->

## Overview

The policy predicts discrete action tokens and refines the resulting action
sequence with a continuous branch. The two branches use separate action tokens
and share the vision-language backbone. The released training configuration and
evaluation protocol are documented in [LIBERO](docs/LIBERO.md).

## Getting Started

Follow the [LIBERO setup, training, and evaluation guide](docs/LIBERO.md).
Datasets and pretrained weights are downloaded separately; no checkpoints are
included in this repository.

## TODO

- [x] Release LIBERO training and evaluation code.
- [ ] Add RoboTwin training and evaluation code.

## Citation

```bibtex
@article{wang2026discrete,
  title={Discrete Forcing: Infusing Discrete Guidance into Continuous Denoising for Few-Step Action Experts},
  author={Wang, Jingbo and others},
  journal={arXiv preprint arXiv:2609.39526},
  year={2026}
}
```

## Acknowledgments

This implementation builds on [StarVLA](https://github.com/starVLA/starVLA).
We thank its authors and contributors for releasing the codebase.

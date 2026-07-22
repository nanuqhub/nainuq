# Inference API

This page is auto-generated from the docstrings in `src/inference/`.

---

## test.py

Entry point for running evaluation of a trained NaiNUQ checkpoint.

::: inference.test
    options:
      members:
        - load_hyperparameters
        - str2bool
        - parse_args
        - main

---

## test_utils.py

Contains the `Test` class that handles the autoregressive rollout,
normalization, post-processing, and metric computation.

::: inference.test_utils
    options:
      members:
        - Test

---

## compute_metrics.py

Standalone metric utilities (RMSE, bias, persistence score).

::: inference.compute_metrics

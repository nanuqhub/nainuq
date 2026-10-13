#!/usr/bin/env python3
"""Run NaiNUQ 6 h emulator inference locally (MacBook / CPU-MPS).

Adapted from ``notebooks/nainuq_demo_6h_rmse.ipynb`` for headless execution.
Requires:
  - ``weights/last.ckpt``        (HuggingFace cdurand95/nainuq)
  - ``data/data_20*.tfrecords.*``(Zenodo test subset)
  - ``data/sea_ice_*_input/output.npy`` (normalisation constants, shipped in repo)
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "nainuq"
WEIGHTS_DIR = REPO_ROOT / "weights"
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "out_local"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(CODE_DIR))

VARIABLES = ["sit", "sic", "siu", "siv", "snt"]
TIMESTEP = 6
NN_SIZE = 32
USE_OCEAN_AS_FORCINGS = True
POST_PROCESSING = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def quiet_np_save():
    # apply_emulator writes a large x_truth.npy at every call; disable it locally.
    with mock.patch.object(np, "save", lambda *a, **k: None), contextlib.redirect_stdout(io.StringIO()):
        yield


def main():
    parser = argparse.ArgumentParser(description="Run NaiNUQ 6h emulator locally")
    parser.add_argument("--k", type=int, default=41, help="emulator steps per forecast (41 = 10 days)")
    parser.add_argument("--frequency", type=int, default=12, help="hours between forecast starts")
    parser.add_argument("--n_cycle", type=int, default=8, help="number of forecasts to run")
    args = parser.parse_args()

    K, FREQUENCY, N_CYCLE = args.k, args.frequency, args.n_cycle

    # --- weights ---
    ckpts = sorted(WEIGHTS_DIR.rglob("*.ckpt")) + sorted(WEIGHTS_DIR.rglob("*.pt"))
    assert ckpts, f"No checkpoint found under {WEIGHTS_DIR}"
    ckpt_path = ([c for c in ckpts if c.name == "last.ckpt"] or ckpts)[0]

    # --- data path (folder holding the normalisation constants) ---
    norm = sorted(DATA_DIR.rglob("sea_ice_mean_input.npy"))
    assert norm, "sea_ice_mean_input.npy not found under data/"
    data_path = str(norm[0].parent) + "/"
    tfrecords = sorted(Path(data_path).rglob("data_20*.tfrecords.*"))
    assert tfrecords, "No data_20*.tfrecords.* found under data/"

    device = get_device()
    print(f"[run_local] device={device}  checkpoint={ckpt_path.name}")
    print(f"[run_local] data_path={data_path}  shards={len(tfrecords)}")
    print(f"[run_local] K={K} frequency={FREQUENCY} N_CYCLE={N_CYCLE}")

    from layers.full_UNet import UNetModel
    from inference.test_utils import Test

    N_OCEAN = 5 if USE_OCEAN_AS_FORCINGS else 0
    N_UNDER = 0
    IN_CH = 7 + N_UNDER + N_OCEAN + len(VARIABLES)  # 17
    OUT_CH = len(VARIABLES)

    model = UNetModel(
        data_path=data_path, in_channels=IN_CH, out_channels=OUT_CH, base_features=NN_SIZE,
        lr=1e-4, weight_decay=1e-3, lambda_bias=0, lambda_PINN=0, lambda_TV=0,
        save_dir=str(OUT_DIR),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- read TFRecords (hourly) ---
    N_TOT_INPUT = 17 if USE_OCEAN_AS_FORCINGS else 12
    steps_per_cycle = (K - 1) * TIMESTEP + 1
    n_needed = (N_CYCLE - 1) * FREQUENCY + steps_per_cycle
    feature_description = {
        "inputs": tf.io.FixedLenFeature([N_TOT_INPUT * 128 * 128], tf.float32),
        "outputs": tf.io.FixedLenFeature([5 * 128 * 128], tf.float32),
    }
    ds = tf.data.TFRecordDataset([str(f) for f in tfrecords]).take(n_needed)
    records = []
    for raw in ds:
        parsed = tf.io.parse_single_example(raw, feature_description)
        records.append(parsed["inputs"].numpy().reshape(N_TOT_INPUT, 128, 128))
    x_all = torch.nan_to_num(torch.from_numpy(np.stack(records)).float(), 0.0)
    del records
    print(f"[run_local] loaded {tuple(x_all.shape)} = (hours, channels, y, x)")

    n_max = (x_all.shape[0] - steps_per_cycle) // FREQUENCY + 1
    if N_CYCLE > n_max:
        print(f"[run_local] only {x_all.shape[0]} records -> N_CYCLE reduced to {n_max}")
        N_CYCLE = n_max

    assert x_all.shape[1] == IN_CH, (x_all.shape[1], IN_CH)
    with torch.no_grad():
        y = model(x_all[:1].to(device))
    print(f"[run_local] one-step output shape: {tuple(y.shape)} (expected (1, 5, 128, 128))")

    run_args = argparse.Namespace(sea_ice_variables=VARIABLES, k=K)
    tester = Test(
        run_args, model=model, frequency=FREQUENCY,
        use_ocean_as_forcings=USE_OCEAN_AS_FORCINGS, N_ocean=N_OCEAN,
        post_processing=POST_PROCESSING, N_under=N_UNDER,
        N_inputs=IN_CH, N_outputs=OUT_CH, ocean=False, season="all",
        k=K, N_cycle=N_CYCLE, timestep=TIMESTEP, save_pred=False,
        path_to_save=str(OUT_DIR) + "/", path_to_data=data_path,
        noise_init=False, noise=0,
    )

    fs, fs_pers, bias = [], [], []
    pred0 = truth0 = None
    for i in range(N_CYCLE):
        start = i * FREQUENCY
        window = x_all[start: start + steps_per_cycle]
        with quiet_np_save():
            rmse, rmse_pers, b, pred, truth = tester.apply_emulator(window)
        fs.append(rmse); fs_pers.append(rmse_pers); bias.append(b)
        if i == 0:
            pred0, truth0 = pred, truth
        print(f"[run_local] forecast {i+1}/{N_CYCLE} done")

    fs, fs_pers, bias = map(np.stack, (fs, fs_pers, bias))
    np.save(OUT_DIR / "demo_fs_mean.npy", fs)
    np.save(OUT_DIR / "demo_fs_mean_pers.npy", fs_pers)
    np.save(OUT_DIR / "demo_bias_mean.npy", bias)
    np.save(OUT_DIR / "demo_pred0.npy", pred0)
    np.save(OUT_DIR / "demo_truth0.npy", truth0)

    print(f"[run_local] saved results to {OUT_DIR}/")
    lead_days = np.arange(K) * TIMESTEP / 24
    units = {"sit": "m", "sic": "-", "siu": "m/s", "siv": "m/s", "snt": "m"}
    for lead in (1, 5, 10):
        idx = int(round(lead * 24 / TIMESTEP))
        if idx >= K:
            continue
        print(f"--- lead {lead} d (step {idx}) ---")
        for j, v in enumerate(VARIABLES):
            print(f"  {v:<5} ({units[v]:<4}) emulator={fs[:, idx, j].mean():.4f}  pers={fs_pers[:, idx, j].mean():.4f}")


if __name__ == "__main__":
    main()

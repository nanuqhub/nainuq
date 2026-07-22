#!/usr/bin/env python3


import argparse
import numpy as np
import torch
from tqdm import trange
import os
import sys
import json

# Add the parent directory to the path so we can import from layers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.full_UNet import UNetModel
from layers.full_UNet_with_PConv import PConv_UNetModel
from inference.test_utils import Test


def load_hyperparameters(save_dir):
    """
    Load training hyperparameters from a saved experiment directory.
 
    Looks for a ``config.json`` file written during training. Falls back
    to sensible defaults if the file is absent.
 
    Parameters
    ----------
    save_dir : str
        Path to the experiment directory (same directory that contains
        the ``checkpoints/`` sub-folder).
 
    Returns
    -------
    dict
        Dictionary with keys ``in_channels``, ``out_channels``,
        ``base_features``, ``lr``, ``weight_decay``, ``lambda_``.
    """
    config_path = os.path.join(save_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    else:
        # Default parameters if config doesn't exist
        return {
            "in_channels": 17,
            "out_channels": 5,
            "base_features": 32,
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "lambda_": 100,
        }


def str2bool(v):
    """
    Convert a string argument to a boolean.
 
    Accepts ``yes / true / t / 1`` as ``True`` and
    ``no / false / f / 0`` as ``False`` (case-insensitive).
 
    Parameters
    ----------
    v : bool or str
        Value to convert.
 
    Returns
    -------
    bool
 
    Raises
    ------
    argparse.ArgumentTypeError
        If the string cannot be interpreted as a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    """
    Parse command-line arguments for the inference script.
 
    Returns
    -------
    argparse.Namespace
        Parsed arguments. Key fields:
 
        - ``save_dir`` : experiment directory (checkpoints + results)
        - ``data_path`` : root directory of the test TFRecord files
        - ``checkpoint_name`` : filename of the ``.pt`` checkpoint
        - ``n_cycle`` : number of independent forecast cycles to evaluate
        - ``frequency`` : time-step offset between consecutive cycles
        - ``k`` : forecast horizon (number of emulator steps)
        - ``timestep`` : temporal resolution in hours (1, 6, 12, or 24)
        - ``sea_ice_variables`` : list of variable names to emulate
          (subset of ``['sit', 'sic', 'siu', 'siv', 'snt']``)
        - ``use_pconv`` : use PConv-UNet instead of standard UNet
        - ``post_processing`` : apply physical post-processing mask
        - ``ocean`` : also evaluate ocean variables
        - ``noise`` : standard deviation of inference-time noise
    """
    parser = argparse.ArgumentParser(
        description="Test UNet model for sea ice prediction"
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True)
    parser.add_argument("--n_cycle", type=int, required=True)
    parser.add_argument("--frequency", type=int, required=True)
    parser.add_argument("--NN_size", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--save_pred", type=str2bool, default=True)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--ocean", type=str2bool, default=True)
    parser.add_argument("--noise_init", type=bool, default=True)
    parser.add_argument("--use_pconv", type=str2bool, default=False)
    parser.add_argument(
        "--sea_ice_variables",
        nargs="+",
        default=["sit", "sic", "siu", "siv", "snt"],
        help="List of sea ice names",
    )
    parser.add_argument("--use_ocean_as_forcings", type=str2bool, default=False)
    parser.add_argument("--ocean_under", type=str2bool, default=False)
    parser.add_argument("--ocean_variables", type=str2bool, default=False)
    parser.add_argument("--post_processing", type=str2bool, default=True)
    return parser.parse_args()


def main():
    """
    Main inference routine.
 
    Workflow
    --------
    1. Parse arguments and derive channel counts.
    2. Load hyperparameters from ``config.json`` in ``save_dir``.
    3. Instantiate the model (``UNetModel`` or ``PConv_UNetModel``).
    4. Load the requested checkpoint.
    5. Instantiate :class:`~inference.test_utils.Test` and call
       :meth:`~inference.test_utils.Test.test_model`.
    6. Save RMSE, bias, and persistence arrays to ``save_dir/test_results/``.
    """
    # Read arguments
    args = parse_args()

    # Define number of features
    len_variables = len(args.sea_ice_variables)
    N_ocean = 0
    N_under = 0

    # Add ocean features if necessary
    if args.use_ocean_as_forcings == True:
        N_ocean = 5
    else:
        N_ocean = 0

    # Add ocean 1st layer if necessary
    if args.ocean_under == False:
        N_under = 0
    else:
        N_under = 2

    # Define input and output size of the emulator
    in_channels = 7 + N_under + N_ocean + len_variables
    out_channels = len_variables

    # Add outputs channels if learning the ocean
    if args.ocean_variables:
        out_channels += 5

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process mask
    # os.chdir('/bettik/PROJECTS/pr-sasip/ducharlo/nanuk/')

    # Load hyperparameters from experiment directory
    config = load_hyperparameters(args.save_dir)

    # Initialize model with loaded hyperparameters
    if args.use_pconv:
        model = PConv_UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=args.NN_size,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            lambda_bias=0,
            lambda_TV=0,
            lambda_PINN=0,
            save_dir=args.save_dir,
        ).to(device)
    else:
        model = UNetModel(
            data_path=args.data_path,
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=args.NN_size,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            lambda_bias=0,
            lambda_PINN=0,
            lambda_TV=0,
            save_dir=args.save_dir,
        ).to(device)
    # After model definition, before loading checkpoint
    # if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs")
    #    model = torch.nn.DataParallel(model)
    # model = model.cuda()
    # Load weights
    checkpoint_path = os.path.join(args.save_dir, "checkpoints", args.checkpoint_name)
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # model = model.cuda()
    # Initialize test class
    test = Test(
        args,
        model=model,
        use_ocean_as_forcings=args.use_ocean_as_forcings,
        N_ocean=N_ocean,
        frequency=args.frequency,
        post_processing=args.post_processing,
        N_under=N_under,
        N_inputs=in_channels,
        N_outputs=out_channels,
        ocean=args.ocean,
        season="all",
        k=args.k,
        N_cycle=args.n_cycle,
        save_pred=args.save_pred,
        timestep=args.timestep,
        path_to_save=args.save_dir,
        path_to_data=args.data_path,
        noise=args.noise,
        noise_init=args.noise_init,
    )

    # Run test
    if args.ocean:
        fs, fs_pers, bias, _, _, fs_oce, fs_pers_oce, bias_oce, _, _ = test.test_model()
    else:
        fs, fs_pers, bias, _, _ = test.test_model()
    # Save results
    results_dir = os.path.join(args.save_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    if args.post_processing:
        np.save(
            os.path.join(
                results_dir, f"test_post_2_process_cycle_{args.n_cycle}_bias_mean.npy"
            ),
            bias,
        )
        np.save(
            os.path.join(
                results_dir, f"test_post_2_process_cycle_{args.n_cycle}_fs_mean.npy"
            ),
            fs,
        )
        np.save(
            os.path.join(
                results_dir,
                f"test_post_2_process_cycle_{args.n_cycle}_fs_mean_pers.npy",
            ),
            fs_pers,
        )
        if args.ocean:
            np.save(
                os.path.join(
                    results_dir,
                    f"post_process_cycle_{args.n_cycle}_bias_mean_ocean.npy",
                ),
                bias_oce,
            )
            np.save(
                os.path.join(
                    results_dir, f"post_process_cycle_{args.n_cycle}_fs_mean_ocean.npy"
                ),
                fs_oce,
            )
            np.save(
                os.path.join(
                    results_dir,
                    f"post_process_cycle_{args.n_cycle}_fs_mean_pers_ocean.npy",
                ),
                fs_pers_oce,
            )
    else:
        np.save(
            os.path.join(results_dir, f"clip_cycle_{args.n_cycle}_bias_mean.npy"), bias
        )
        np.save(os.path.join(results_dir, f"clip_cycle_{args.n_cycle}_fs_mean.npy"), fs)
        np.save(
            os.path.join(results_dir, f"clip_cycle_{args.n_cycle}_fs_mean_pers.npy"),
            fs_pers,
        )
        if args.ocean:
            np.save(
                os.path.join(
                    results_dir,
                    f"post_process_cycle_{args.n_cycle}_bias_mean_ocean.npy",
                ),
                bias_oce,
            )
            np.save(
                os.path.join(
                    results_dir, f"post_process_cycle_{args.n_cycle}_fs_mean_ocean.npy"
                ),
                fs_oce,
            )
            np.save(
                os.path.join(
                    results_dir,
                    f"post_process_cycle_{args.n_cycle}_fs_mean_pers_ocean.npy",
                ),
                fs_pers_oce,
            )


if __name__ == "__main__":
    main()

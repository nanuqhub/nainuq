#!/usr/bin/env python
"""CLI to build the NaiNUQ 1h Zarr dataset from NANUQ + ERA5.

Example
-------
python build_dataset.py \
    --sim_dir /summer/meom/MEOM-OPENDAP/simus_nanuq/NANUK1/NANUK1-N1CPL00-S \
    --atmo_forcings_dir /summer/sasip/model-forcings/atmo_forcing/ERA5_full \
    --save_dir /summer/meom/workdir/ducharlo/dataset/nanuk1_1h_zarr \
    --train 2010 2017 --val 2018 2018 --test 2019 2020 \
    --build --normalize

This replaces the old ``data_to_tfrecords`` notebook. The output is a set of
consolidated Zarr stores (``train.zarr``, ``val.zarr``, ``test.zarr``) plus a
``normalization.zarr`` with the training mean/std statistics.
"""

from __future__ import annotations

import argparse

from nainuq.datapipeline import Config, Splits
from nainuq.datapipeline.build import build_all
from nainuq.datapipeline.normalization import build_normalization


def parse_args():
    p = argparse.ArgumentParser(description="Build the NaiNUQ 1h Zarr dataset")
    p.add_argument("--sim_dir", type=str, required=True)
    p.add_argument("--atmo_forcings_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--train", nargs=2, type=int, default=[2010, 2017])
    p.add_argument("--val", nargs=2, type=int, default=[2018, 2018])
    p.add_argument("--test", nargs=2, type=int, default=[2019, 2020])
    p.add_argument("--months_begin", type=int, default=1)
    p.add_argument("--months_end", type=int, default=12)
    p.add_argument("--sea_ice_variables", nargs="+",
                   default=["sit", "sic", "siu", "siv", "snt"])
    p.add_argument("--atmo_variables", nargs="+",
                   default=["t2m", "mtpr", "d2m", "msdswrf", "msdlwrf"])
    p.add_argument("--wind_variables", nargs="+", default=["u10", "v10"])
    p.add_argument("--ocean_variables", nargs="+", default=["zos", "tos", "sos"])
    p.add_argument("--ocean_forcings", nargs="+", default=["vozocrtx", "vomecrty"])
    p.add_argument("--use_ocean", action="store_true", default=True)
    p.add_argument("--no-ocean", dest="use_ocean", action="store_false")
    p.add_argument("--time_chunk", type=int, default=720)
    p.add_argument("--atmo_file_pattern", type=str, default="ERA5_hourly_{var}_y{year}.nc")
    p.add_argument("--build", action="store_true", help="Build train/val/test stores")
    p.add_argument("--normalize", action="store_true", help="Compute normalization stats")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        sim_dir=args.sim_dir,
        atmo_forcings_dir=args.atmo_forcings_dir,
        save_dir=args.save_dir,
        splits=Splits(
            train=tuple(args.train),
            val=tuple(args.val),
            test=tuple(args.test),
        ),
        months_begin=args.months_begin,
        months_end=args.months_end,
        sea_ice_variables=args.sea_ice_variables,
        atmo_variables=args.atmo_variables,
        wind_variables=args.wind_variables,
        ocean_variables=args.ocean_variables,
        ocean_forcings=args.ocean_forcings,
        use_ocean=args.use_ocean,
        time_chunk=args.time_chunk,
        atmo_file_pattern=args.atmo_file_pattern,
    )
    if args.build:
        build_all(cfg)
    if args.normalize:
        build_normalization(cfg)
    print("Done.")


if __name__ == "__main__":
    main()

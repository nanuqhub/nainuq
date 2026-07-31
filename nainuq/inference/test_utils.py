from pathlib import Path
import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader
import sys
import os

# Add the parent directory to the path so we can import from datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.TFRecordDataset import Sea_ice_dataset  # Direct import


class Test:
    def __init__(
        self,
        args,
        model,
        frequency,
        use_ocean_as_forcings,
        N_ocean,
        post_processing,
        N_under,
        N_inputs,
        N_outputs,
        ocean,
        season,
        k,
        N_cycle,
        timestep,
        save_pred,
        path_to_save,
        path_to_data,
        noise_init,
        noise,
    ):
        self.model = model
        self.ocean = ocean
        self.use_ocean_as_forcings = use_ocean_as_forcings
        self.frequency = frequency
        self.N_ocean = N_ocean
        self.N_under = N_under
        self.post_processing = post_processing
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        self.path = path_to_save
        self.timestep = timestep
        self.N_cycle = N_cycle
        self.timestep_output = 2
        self.path_data = path_to_data
        self.noise_init = noise_init
        self.noise = noise
        self.save_pred = save_pred
        self.device = next(model.parameters()).device

        self.mask = np.load(
            "/linkhome/rech/genrea01/ucm13rr/nanuq1/src/mask2_nanuk1.npy"
        )
        # self.mask = torch.from_numpy(self.mask).float()
        # self.mask = self.mask.permute(2, 0, 1).unsqueeze(0)
        # self.mask = torch.reshape(self.mask, [1, 1, 128, 128])
        # Move mask to GPU if available
        # if torch.cuda.is_available():
        #   print("Moving mask to GPU")
        #  self.mask = self.mask.cuda()
        self.sea_ice_variables = args.sea_ice_variables
        # Ocean normalization constants
        if self.ocean:
            self.mean_input_ocean = np.load(f"{self.path_data}ocean_mean_input.npy")
            self.mean_output_ocean = np.load(f"{self.path_data}ocean_mean_output.npy")
            self.std_input_ocean = np.load(f"{self.path_data}ocean_std_input.npy")
            self.std_output_ocean = np.load(f"{self.path_data}ocean_std_output.npy")

        # Sea ice normalization constants
        self.mean_input = np.load(f"{self.path_data}sea_ice_mean_input.npy")
        self.std_input = np.load(f"{self.path_data}sea_ice_std_input.npy")
        self.mean_output = np.load(f"{self.path_data}sea_ice_mean_output.npy")
        self.std_output = np.load(f"{self.path_data}sea_ice_std_output.npy")

        # Number of grid cells
        self.N_x = 128
        self.N_y = 128

        # Number of emulator iteration
        self.k = args.k

        # Number of sea ice variables emulated
        self.N = len(self.sea_ice_variables)

        # Index of SIC position
        self.N_sic = 1

        # Definition of minimal h_min and c_min for postprocessing step (cf neXtSIM code)
        if "sit" in self.sea_ice_variables:
            self.hmin = (1e-5 - self.mean_input[0]) / self.std_input[0]
            # self.hmin = 0.1*self.hmin
            self.h0 = -self.mean_input[0] / self.std_input[0]
        if "sic" in self.sea_ice_variables:
            self.cmin = (0.0001 - self.mean_input[1]) / self.std_input[1]
            self.c0 = -self.mean_input[1] / self.std_input[1]

        # Selection of normalization constant if SIT only is learnt
        if self.sea_ice_variables == ["sit"]:
            self.mean_input = self.mean_input[0]
            self.mean_output = self.mean_output[0]
            self.std_input = self.std_input[0]
            self.std_output = self.std_output[0]

        # Selection of normalization constant if SIC only is learnt
        if self.sea_ice_variables == ["sic"]:
            self.N_sic = 0
            self.mean_input = self.mean_input[1]
            self.mean_output = self.mean_output[1]
            self.std_input = self.std_input[1]
            self.std_output = self.std_output[1]
        # Selection of normalization constant if SIT and SIC are learnt
        if self.sea_ice_variables == ["sit", "sic"]:
            self.N_sic = 1
            self.mean_input = self.mean_input[:2]
            self.mean_output = self.mean_output[:2]
            self.std_input = self.std_input[:2]
            self.std_output = self.std_output[:2]
        # Selection of normalization constant if SIT and SIC are learnt
        if self.sea_ice_variables == ["siu", "siv"]:
            self.mean_input = self.mean_input[2:4]
            self.mean_output = self.mean_output[2:4]
            self.std_input = self.std_input[2:4]
            self.std_output = self.std_output[2:4]

        self.norm = -self.mean_input / self.std_input
        self.norm = self.norm.reshape((self.N, 1, 1))

    def add_noise(self, x):
        return x + torch.randn_like(x) * self.noise

    # Define normalization function for sea ice input
    def normalize_input(self, x):
        std = self.std_input.reshape(1, self.N, 1, 1)
        mean = self.mean_input.reshape(1, self.N, 1, 1)
        return (x - mean) / std

        # Define normalization function for ocean input       # Number of sea ice variables emulated
        self.N = len(self.sea_ice_variables)

        # Index of SIC position
        self.N_sic = 1

        # Definition of minimal h_min and c_min for postprocessing step (cf neXtSIM code)
        if "sit" in self.sea_ice_variables:
            self.hmin = (1e-5 - self.mean_input[0]) / self.std_input[0]
            # self.hmin = 0.1*self.hmin
            self.h0 = -self.mean_input[0] / self.std_input[0]
        elif "sic" in self.sea_ice_variables:
            self.cmin = (1e-4 - self.mean_input[1]) / self.std_input[1]
            self.c0 = -self.mean_input[1] / self.std_input[1]

        # Selection of normalization constant if SIT only is learnt
        if self.sea_ice_variables == ["sit"]:
            self.mean_input = self.mean_input[0]
            self.mean_output = self.mean_output[0]
            self.std_input = self.std_input[0]
            self.std_output = self.std_output[0]

        # Selection of normalization constant if SIC only is learnt
        if self.sea_ice_variables == ["sic"]:
            self.N_sic = 0
            self.mean_input = self.mean_input[1]
            self.mean_output = self.mean_output[1]
            self.std_input = self.std_input[1]
            self.std_output = self.std_output[1]
        # Selection of normalization constant if SIT and SIC are learnt
        if self.sea_ice_variables == ["sit", "sic"]:
            self.N_sic = 1
            self.mean_input = self.mean_input[:2]
            self.mean_output = self.mean_output[:2]
            self.std_input = self.std_input[:2]
            self.std_output = self.std_output[:2]
        # Selection of normalization constant if SIT and SIC are learnt
        if self.sea_ice_variables == ["siu", "siv"]:
            self.mean_input = self.mean_input[2:4]
            self.mean_output = self.mean_output[2:4]
            self.std_input = self.std_input[2:4]
            self.std_output = self.std_output[2:4]

        self.norm = -self.mean_input / self.std_input
        self.norm = self.norm.reshape((self.N, 1, 1))

    def add_noise(self, x):
        return x + torch.randn_like(x) * self.noise

    # Define normalization function for sea ice input
    def normalize_input(self, x):
        std = self.std_input.reshape(1, self.N, 1, 1)
        mean = self.mean_input.reshape(1, self.N, 1, 1)
        return (x - mean) / std

    def normalize_input_ocean(self, x):
        std = self.std_input_ocean.reshape(1, self.N, 1, 1)
        mean = self.mean_input_ocean.reshape(1, self.N, 1, 1)
        return (x - mean) / std

    # Define reverse normalization function for sea ice output
    def reverse_normalize_output(self, x):
        std = self.std_output.reshape(1, self.N, 1, 1)
        mean = self.mean_output.reshape(1, self.N, 1, 1)
        return x * std + mean

    # Define reverse normalization function for ocean output
    def reverse_normalize_output_ocean(self, x):
        std = self.std_output_ocean.reshape(1, self.N, 1, 1)
        mean = self.mean_output_ocean.reshape(1, self.N, 1, 1)
        return x * std + mean

    # Define reverse normalization function for sea ice input
    def reverse_normalize_input(self, x):
        std = self.std_input.reshape(1, self.N, 1, 1)
        mean = self.mean_input.reshape(1, self.N, 1, 1)
        return x * std + mean

    # Define reverse normalization function for ocean input
    def reverse_normalize_input_ocean(self, x):
        std = self.std_input_ocean.reshape(1, self.N, 1, 1)
        mean = self.mean_input_ocean.reshape(1, self.N, 1, 1)

        return x * std + mean

    def apply_emulator(self, x_truth):

        x_truth = x_truth.numpy().astype(np.float32)
        np.save(f"{self.path}x_truth.npy", x_truth)
        # Keep min and max values for clipping
        min_truth = np.min(x_truth[0, : self.N, :, :], axis=(1, 2)).reshape(
            (self.N, 1, 1)
        )
        max_truth = np.max(x_truth[0, : self.N, :, :], axis=(1, 2)).reshape(
            (self.N, 1, 1)
        )
        min_truth[2:4] = min_truth[2:4] - 0.3
        max_truth[2:4] = max_truth[2:4] + 0.3
        # Initialize results storage
        result = np.zeros((self.k))
        x_inputs = np.zeros((self.k, self.N_inputs, self.N_x, self.N_y))
        x_pred_SI = np.zeros((self.k, self.N, self.N_x, self.N_y))
        if self.ocean:
            x_pred_ocean = np.zeros((self.k, self.N, self.N_x, self.N_y))

        # Initialize first step (truth from sea ice model)
        x_inputs[0] = x_truth[0]
        x_pred_SI[0] = x_truth[0, : self.N, :, :]
        if self.ocean:
            x_pred_ocean[0] = x_truth[0, self.N + 7 :, :, :]  # TO CHECK

        # Apply the emulator
        for t in range(1, self.k):

            # Convert to tensor and move to device
            inputs = (
                torch.from_numpy(x_inputs[t - 1]).float().unsqueeze(0).to(self.device)
            )

            # Get prediction
            with torch.no_grad():

                # Apply emulator to get the increment
                err = self.model(inputs)[0]

                # Add to previous state to get the state at t + 1 for sea ice
                err_SI = err[: self.N]
                err_SI = err_SI.cpu().numpy()  # Shape is now [C, H, W]
                err_SI = self.reverse_normalize_output(err_SI)
                x_pred_SI[t] = self.reverse_normalize_input(x_pred_SI[t - 1]) + err_SI
                # np.save(f'{self.path}x_pred.npy',x_pred_SI)
                x_pred_SI[t] = self.normalize_input(x_pred_SI[t])
                # x_pred_SI[t] = np.clip(x_pred_SI[t], a_min=min_truth, a_max=max_truth)

                # Post processing step
                if self.post_processing:
                    if "sit" in self.sea_ice_variables:
                        print(np.sum([x_pred_SI[t, 0] > self.hmin][0]))
                        x_pred_SI[t] = np.where(
                            x_pred_SI[t, 0] > self.hmin, x_pred_SI[t], self.norm
                        )
                    elif "sic" in self.sea_ice_variables:
                        print(np.sum([x_pred_SI[t, 1] > self.cmin][0]))
                        x_pred_SI[t] = np.where(
                            x_pred_SI[t, self.N_sic] > self.cmin,
                            x_pred_SI[t],
                            self.norm,
                        )
                    else:
                        x_pred_SI[t] = x_pred_SI[t]
                    x_pred_SI[t] = (
                        self.reverse_normalize_input(x_pred_SI[t]) * self.mask
                    )
                    x_pred_SI[t] = self.normalize_input(x_pred_SI[t])
                x_pred_SI[t] = np.clip(x_pred_SI[t], a_min=min_truth, a_max=max_truth)
                # Add to previous state to get the state at t + 1 for ocean
                if self.ocean:
                    err_ocean = err[self.N :]
                    err_ocean = err_ocean.cpu().numpy()
                    err_ocean = self.reverse_normalize_output(err_ocean)
                    x_pred_ocean[t] = (
                        self.reverse_normalize_input(x_pred_ocean[t - 1]) + err_ocean
                    )
                    x_pred_ocean[t] = self.normalize_input(x_pred_ocean[t])

            # Add corresponding forcings
            if self.ocean:
                x_inputs[t] = np.concatenate(
                    (
                        x_pred_SI[t],  # Current prediction
                        x_truth[
                            t * self.timestep, self.N : self.N + 9, :, :
                        ],  # TO CHECK
                        x_pred_ocean[t],
                    ),  # Future forcings
                    axis=0,
                )
            else:
                x_inputs[t] = np.concatenate(
                    (
                        x_pred_SI[t],  # Current prediction
                        x_truth[t * self.timestep, self.N : self.N_inputs, :, :],
                    ),  # Future forcings
                    axis=0,
                )

        ### Calculate metrics

        # Denormalize to compute metrics in physical space
        x_pred_squeezed_SI = x_pred_SI[:, : self.N, :, :]
        x_pred_squeezed_SI = (
            self.reverse_normalize_input(x_pred_squeezed_SI) * self.mask
        )
        x_truth_target_SI = x_truth[
            0 : self.k * self.timestep : self.timestep, : self.N, :, :
        ]
        x_truth_target_SI = self.reverse_normalize_input(x_truth_target_SI) * self.mask

        # Calculate bias for each timestep (k steps)
        bias_SI = np.zeros((self.k, self.N))
        for t in range(self.k):
            bias_SI[t] = (
                x_pred_squeezed_SI[t] * self.mask - x_truth_target_SI[t] * self.mask
            ).mean(axis=(1, 2))

        # Calculate RMSE for each timestep
        result_SI = np.zeros((self.k, self.N))
        for t in range(self.k):
            result_SI[t] = np.sqrt(
                (
                    (
                        x_pred_squeezed_SI[t] * self.mask
                        - x_truth_target_SI[t] * self.mask
                    )
                    ** 2
                ).mean(axis=(1, 2))
            )

        # Calculate persistence RMSE
        result_pers_SI = np.zeros((self.k, self.N))
        for t in range(self.k):
            result_pers_SI[t] = np.sqrt(
                (
                    (
                        x_truth_target_SI[0] * self.mask
                        - x_truth_target_SI[t] * self.mask
                    )
                    ** 2
                ).mean(axis=(1, 2))
            )

        if self.ocean:
            x_pred_squeezed_ocean = x_pred_ocean[:, : self.N, :, :]
            x_pred_squeezed_ocean = self.reverse_normalize_input_ocean(
                x_pred_squeezed_ocean
            )
            x_truth_target_ocean = x_truth[
                0 : self.k * self.timestep : self.timestep, self.N + 9 :, :, :
            ]
            x_truth_target_ocean = self.reverse_normalize_input_ocean(
                x_truth_target_ocean
            )
            # Calculate bias for each timestep (k steps)
            bias_ocean = np.zeros((self.k, self.N))
            for t in range(self.k):
                bias_ocean[t] = (
                    x_pred_squeezed_ocean[t] - x_truth_target_ocean[t]
                ).mean(axis=(1, 2))
            # Calculate RMSE for each timestep
            result_ocean = np.zeros((self.k, self.N))
            for t in range(self.k):
                result_ocean[t] = np.sqrt(
                    ((x_pred_squeezed_ocean[t] - x_truth_target_ocean[t]) ** 2).mean(
                        axis=(1, 2)
                    )
                )
            # Calculate persistence RMSE
            result_pers_ocean = np.zeros((self.k, self.N))
            for t in range(self.k):
                result_pers_ocean[t] = np.sqrt(
                    ((x_truth_target_ocean[0] - x_truth_target_ocean[t]) ** 2).mean(
                        axis=(1, 2)
                    )
                )

        if self.ocean:
            return (
                result_SI,
                result_pers_SI,
                bias_SI,
                x_pred_squeezed_SI,
                x_truth_target_SI,
                result_ocean,
                result_pers_ocean,
                bias_ocean,
                x_pred_squeezed_ocean,
                x_truth_target_ocean,
            )
        else:
            return (
                result_SI,
                result_pers_SI,
                bias_SI,
                x_pred_squeezed_SI,
                x_truth_target_SI,
            )

    def test_model(self):
        all_years = [2019, 2020]  # toutes les années disponibles

        # Initialize metrics
        fs_SI = np.zeros((self.N_cycle, self.k, self.N))
        bias_SI = np.zeros((self.N_cycle, self.k, self.N))
        fs_pers_SI = np.zeros((self.N_cycle, self.k, self.N))
        x_pred_SI = np.zeros((self.N_cycle, self.k, self.N_x, self.N_y, self.N))

        if self.ocean:
            fs_ocean = np.zeros((self.N_cycle, self.k, self.N))
            bias_ocean = np.zeros((self.N_cycle, self.k, self.N))
            fs_pers_ocean = np.zeros((self.N_cycle, self.k, self.N))
            x_pred_ocean = np.zeros((self.N_cycle, self.k, self.N_x, self.N_y, self.N))

        def load_year(year):
            N_files = len(
                list(Path(f"{self.path_data}test/").rglob(f"data_{year}.tfrecords.*"))
            )
            files = [
                f"{self.path_data}test/data_{year}.tfrecords.{str(i).zfill(3)}"
                for i in range(N_files)
            ]
            is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            batch_size = 8784 if is_leap else 8760
            print(files)
            dataset = Sea_ice_dataset(
                files,
                self.sea_ice_variables,
                self.N_ocean,
                self.N_under,
                self.use_ocean_as_forcings,
            )

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for x, y in loader:
                print(np.shape(x.cpu().numpy()))
                np.save(f"{self.path}x.npy", x.cpu().numpy())
                print(np.shape(y.cpu().numpy()))
                np.save(f"{self.path}y.npy", y.cpu().numpy())
                return x.cpu()  # ← CPU only, never .to(self.device)

        # Déterminer les années à charger
        # self.frequency = index de départ dans l'année (ex: 4380 = milieu d'année)
        start_year = all_years[0]
        x_current = load_year(start_year)

        # Construire le buffer de données : année courante + suivante si nécessaire
        # pour couvrir self.frequency + self.N_cycle * k * timestep pas de temps
        total_steps_needed = (
            self.frequency * (self.N_cycle - 1) + self.k * self.timestep + 1
        )
        print(total_steps_needed)
        if total_steps_needed > x_current.shape[0]:
            # Le forecast déborde sur l'année suivante
            print("Forecast déborde")
            next_year = all_years[1]
            x_next = load_year(next_year)
            x_full = torch.cat([x_current, x_next], dim=0)
            print(f"Forecast déborde sur {next_year}, buffer concatené: {x_full.shape}")
        else:
            x_full = x_current
            print(f"Forecast contenu dans {start_year}: {x_full.shape}")

        # Découper à partir de self.frequency
        x = x_full
        print(f"Démarrage au timestep {self.frequency}, shape après découpe: {x.shape}")

        self.model.eval()

        for i in trange(self.N_cycle):

            slice_ = x[
                i * self.frequency : i * self.frequency + self.k * self.timestep + 1
            ]
            print(np.shape(slice_))
            if self.ocean:
                (
                    fs_res_SI,
                    fs_pers_res_SI,
                    bias_res_SI,
                    xpred_SI,
                    xtruth_SI,
                    fs_res_ocean,
                    fs_pers_res_ocean,
                    bias_res_ocean,
                    xpred_ocean,
                    xtruth_ocean,
                ) = self.apply_emulator(slice_)
            else:
                (
                    fs_res_SI,
                    fs_pers_res_SI,
                    bias_res_SI,
                    xpred_SI,
                    xtruth_SI,
                ) = self.apply_emulator(slice_)
            # torch.cuda.empty_cache()  # free after each cycle
            # Sauvegarder métriques
            bias_SI[i] = bias_res_SI.reshape((self.k, self.N))
            fs_SI[i] = fs_res_SI.reshape((self.k, self.N))
            fs_pers_SI[i] = fs_pers_res_SI.reshape((self.k, self.N))
            x_pred_SI[i] = np.transpose(xpred_SI, [0, 2, 3, 1])

            if self.ocean:
                bias_ocean[i] = bias_res_ocean.reshape((self.k, self.N))
                fs_ocean[i] = fs_res_ocean.reshape((self.k, self.N))
                fs_pers_ocean[i] = fs_pers_res_ocean.reshape((self.k, self.N))
                x_pred_ocean[i] = np.transpose(xpred_ocean, [0, 2, 3, 1])

            if self.save_pred:
                np.save(
                    f"{self.path}test_process_{self.post_processing}_freq_{self.frequency}_cycle_{self.N_cycle}_truth_{i}.npy",
                    xtruth_SI,
                )
                np.save(
                    f"{self.path}test_process_{self.post_processing}_freq_{self.frequency}_cycle_{self.N_cycle}_pred_{i}.npy",
                    xpred_SI,
                )
                if self.ocean:
                    np.save(
                        f"{self.path}post_process_{self.post_processing}_freq_{self.frequency}_cycle_{self.N_cycle}_truth_ocean{i}.npy",
                        xtruth_ocean,
                    )
                    np.save(
                        f"{self.path}post_process_{self.post_processing}_freq_{self.frequency}_cycle_{self.N_cycle}_pred_ocean{i}.npy",
                        xpred_ocean,
                    )

        N_freq = 24 // self.timestep
        if self.N_cycle > 100:
            x_pred1 = x_pred_SI[:, :N_freq].mean(axis=1)
            np.save(
                f"{self.path}field_{self.post_processing}_freq_{self.frequency}_cycle_{self.N_cycle}_pred_24h.npy",
                x_pred1,
            )
            x_pred10 = x_pred_SI[:, N_freq * 10 : N_freq * 10 + N_freq].mean(axis=1)
            np.save(
                f"{self.path}field_{self.post_processing}_freq_{self.frequency}_cycle_{self.N_cycle}_pred_10d.npy",
                x_pred10,
            )

        return fs_SI, fs_pers_SI, bias_SI, x_pred_SI, xtruth_SI

    # def test_model(self):

    # Setup validation dataset
    # N_file = len(list(Path(f"{self.path_data}test/").rglob('data_20*.tfrecords.*')))
    # N_file = len(list(Path(f"{self.path_data}test/").rglob('data_20*.tfrecords.*')))
    # print(N_file)
    # Load files
    # val_files = [
    #           f"{self.path_data}test/data_20[19,20].tfrecords.{str(i).zfill(3)}" for i in range(N_file)
    #          ]
    # Count the number of TFRecord files for 2019 and 2020
    #   N_file_2019 = len(list(Path(f"{self.path_data}test/").rglob('data_2019.tfrecords.*')))
    #   N_file_2020 = len(list(Path(f"{self.path_data}test/").rglob('data_2020.tfrecords.*')))

    #  N_file = N_file_2019 + N_file_2020
    # print(N_file)

    # Load files for 2019 and 2020
    #   val_files_2019 = []
    #   val_files_2020 = []
    #  for year in ["19"]:
    #       for i in range(N_file_2019):
    #           file_path = f"{self.path_data}test/data_20{year}.tfrecords.{str(i).zfill(3)}"
    #          val_files_2019.append(file_path)
    # for year in ["20"]:
    #    for i in range(N_file_2020):
    #       file_path = f"{self.path_data}test/data_20{year}.tfrecords.{str(i).zfill(3)}"
    #      val_files_2020.append(file_path)
    # val_files = sorted(val_files)
    # val_files = [f"{self.path_data}test/data_20*.tfrecords.{str(i).zfill(3)}" for i in range(N_file)]
    # val_dataset_2019 = Sea_ice_dataset(val_files_2019, self.sea_ice_variables,  self.N_ocean, self.N_under,self.use_ocean_as_forcings)  # Use direct import
    # val_dataset_2020 = Sea_ice_dataset(val_files_2020, self.sea_ice_variables,  self.N_ocean, self.N_under,self.use_ocean_as_forcings)  # Use direct import
    # val_loader = DataLoader(val_dataset, batch_size=8800*2, shuffle=False)
    # val_loader_2019 = DataLoader(val_dataset_2019, batch_size=8760, shuffle=False)
    # val_loader_2020 = DataLoader(val_dataset_2020, batch_size=8640, shuffle=False)

    # Initialize metrics results for sea ice
    # fs_SI = np.zeros((self.N_cycle, self.k, self.N))
    # bias_SI = np.zeros((self.N_cycle, self.k, self.N))
    # fs_pers_SI = np.zeros((self.N_cycle, self.k, self.N))
    # x_pred_SI = np.zeros((self.N_cycle, self.k, self.N_x, self.N_y, self.N))

    # Initialize metrics results for ocean
    # if self.ocean:
    #   fs_ocean = np.zeros((self.N_cycle, self.k, self.N))
    #  bias_ocean = np.zeros((self.N_cycle, self.k, self.N))
    # fs_pers_ocean = np.zeros((self.N_cycle, self.k, self.N))
    # x_pred_ocean = np.zeros((self.N_cycle, self.k, self.N_x, self.N_y, self.N))

    # Get validation data
    # loaders = [val_loader_2019, val_loader_2020]

    # for loader in loaders:
    # flag = 0
    # for x, y in loader:
    # Process each batch (x, y) here
    #   print(f"Processing batch with shape: x={x.shape}, y={y.shape}")
    # Your processing logic goes here
    # x, y = next(iter(val_loader))

    # Remove first example (sometimes 0)
    # if flag==0:
    #    x= x[12:]
    #      y = y[12:]
    # x = x.to(self.device)

    # Move to prediction mode for the NN
    # self.model.eval()

    # Apply the emulator k times for each sample
    # for i in trange(self.N_cycle):
    #    if self.ocean:
    #        fs_res_SI, fs_pers_res_SI, bias_res_SI, xpred_SI, xtruth_SI, fs_res_ocean, fs_pers_res_ocean, bias_res_ocean, xpred_ocean, xtruth_ocean = self.apply_emulator(x[i:i+self.k*self.timestep+1])
    #        bias_SI[i] = bias_res_SI.reshape((self.k, self.N))
    #        fs_SI[i] = fs_res_SI.reshape((self.k, self.N))
    #       fs_pers_SI[i] = fs_pers_res_SI.reshape((self.k, self.N))
    #       x_pred_SI[i] = np.transpose(xpred_SI,[0, 2, 3, 1])
    #     bias_ocean[i] = bias_res_ocean.reshape((self.k, self.N))
    #       fs_ocean[i] = fs_res_ocean.reshape((self.k, self.N))
    #       fs_pers_ocean[i] = fs_pers_res_ocean.reshape((self.k, self.N))
    #       x_pred_ocean[i] = np.transpose(xpred_ocean,[0, 2, 3, 1])
    #   else:
    # print(x[i*self.frequency:i*self.frequency+self.k*self.timestep+1].shape)
    # print(i)
    # print(x[i*self.frequency:i*self.frequency+self.k*self.timestep+1].shape)
    #       fs_res_SI, fs_pers_res_SI, bias_res_SI, xpred_SI, xtruth_SI = self.apply_emulator(x[i*self.frequency:i*self.frequency+self.k*self.timestep+1])
    #      bias_SI[i] = bias_res_SI.reshape((self.k, self.N))
    #     fs_SI[i] = fs_res_SI.reshape((self.k, self.N))
    #    fs_pers_SI[i] = fs_pers_res_SI.reshape((self.k, self.N))
    #   x_pred_SI[i] = np.transpose(xpred_SI,[0, 2, 3, 1])

    # Save prediction
    # if self.save_pred:
    #    tru_SI = xtruth_SI
    #    if self.ocean:
    #        tru_ocean = xtruth_ocean
    #    np.save(f"{self.path}post_2_process_{self.post_processing}_cycle_{self.N_cycle}_truth_{i}.npy", tru_SI)
    #    np.save(f"{self.path}post_2_process_{self.post_processing}_cycle_{self.N_cycle}_pred_{i}.npy", xpred_SI)
    #    if self.ocean:
    #        np.save(f"{self.path}post_process_{self.post_processing}_cycle_{self.N_cycle}_truth_ocean{i}.npy", tru_ocean)
    #        np.save(f"{self.path}post_process_{self.post_processing}_cycle_{self.N_cycle}_pred_ocean{i}.npy", xpred_ocean)
    # N_freq = 24//self.timestep
    # print(N_freq)
    # if self.N_cycle>100:
    #   x_pred1 = x_pred_SI[:,:N_freq].mean(axis=1)
    #    np.save(f"{self.path}field_{self.post_processing}_cycle_{self.N_cycle}_pred_24h.npy", x_pred1)

    #   x_pred10 = x_pred_SI[:,N_freq*10:N_freq*10 + N_freq].mean(axis=1)
    #    np.save(f"{self.path}field_{self.post_processing}_cycle_{self.N_cycle}_pred_10d.npy", x_pred10)
    # return fs_SI, fs_pers_SI, bias_SI, x_pred_SI, xtruth_SI

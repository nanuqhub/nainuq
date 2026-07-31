"""
TFRecord-based PyTorch dataset for NaiNUQ.

Data is stored as TFRecord shards written by a TensorFlow preprocessing
pipeline and loaded on-the-fly into PyTorch tensors. Each record contains
a normalized (inputs, outputs) pair on a 128x128 Arctic grid.

Input channel layout (before variable selection):

====  ===============================
 0    sea-ice thickness       (sit)
 1    sea-ice concentration   (sic)
 2    zonal ice velocity      (siu)
 3    meridional ice velocity (siv)
 4    snow thickness          (snt)
5-11  atmospheric forcings    (7 ch)
12-13 ocean under-ice         (N_under, optional)
14-18 ocean surface forcings  (N_ocean, optional)
====  ===============================

Output channel layout (always 5):
    sit, sic, siu, siv, snt increments.
"""

import tensorflow as tf
import torch
from torch.utils.data import Dataset
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


class Sea_ice_dataset(Dataset):
    """
    PyTorch Dataset wrapping TFRecord shards for sea-ice emulation.

    Reads pre-processed TFRecord files, parses normalised input/output
    pairs, and returns PyTorch float tensors ready for model ingestion.
    NaN values (from the land mask) are replaced with zeros.

    The dataset supports five variable configurations controlled by the
    ``variables`` argument:

    +--------------------+--------------------+------------------+
    | ``variables``      | Input channels     | Output channels  |
    +====================+====================+==================+
    | ``['sit']``        | sit + 7 forcings   | sit increment    |
    +--------------------+--------------------+------------------+
    | ``['sic']``        | sic + 7 forcings   | sic increment    |
    +--------------------+--------------------+------------------+
    | ``['sit','sic']``  | sit,sic + forcings | sit,sic increments|
    +--------------------+--------------------+------------------+
    | ``['siu','siv']``  | siu,siv + forcings | siu,siv increments|
    +--------------------+--------------------+------------------+
    | any other list     | all 5 + forcings   | all 5 increments |
    +--------------------+--------------------+------------------+

    Parameters
    ----------
    filenames : list of str
        Paths to TFRecord shard files
        (e.g. ``['data_2018_jra.tfrecords.000', ...]``).
    variables : list of str
        Sea-ice variable names to emulate. Controls which channels are
        extracted from each record (see table above).
    N_ocean : int
        Number of ocean surface forcing channels (0 or 5).
        Automatically set to 0 when ``use_ocean=False``.
    N_under : int
        Number of ocean under-ice channels (0 or 2).
        Automatically set to 0 when ``use_ocean=False``.
    use_ocean : bool
        If ``True``, ocean forcings are included giving
        ``N_tot_input=17``. If ``False``, only atmospheric forcings
        are used (``N_tot_input=12``).

    Attributes
    ----------
    N_tot_input : int
        Total number of input channels (12 or 17).
    N_tot_output : int
        Total number of output channels (always 5).
    cumulative_sizes : list of int
        Cumulative record counts per shard, used for global index mapping.

    Notes
    -----
    Each ``__getitem__`` call opens the TFRecord file from disk (no
    caching). For large-scale training on HPC, consider increasing
    DataLoader ``num_workers`` or pre-loading shards into RAM.
    """

    def __init__(self, filenames, variables, N_ocean, N_under, use_ocean):
        self.filenames = filenames
        self.variables = variables
        self.N_under = N_under
        self.N_sea_ice = len(self.variables)
        self.use_ocean = use_ocean
        self.N_ocean = N_ocean

        if not self.use_ocean:
            self.N_ocean = 0
            self.N_under = 0
            self.N_tot_input = 12
        else:
            self.N_tot_input = 17

        tf.config.run_functions_eagerly(True)
        print(self.N_tot_input)

        self.N_tot_output = 5

        self.feature_description = {
            "inputs": tf.io.FixedLenFeature([self.N_tot_input * 128 * 128], tf.float32),
            "outputs": tf.io.FixedLenFeature(
                [1 * 128 * 128 * self.N_tot_output], tf.float32
            ),
        }

        print(f"Counting examples in {len(filenames)} files...")
        self.cumulative_sizes = []
        total = 0
        for file in filenames:
            dataset = tf.data.TFRecordDataset(file)
            size = sum(1 for _ in dataset)
            total += size
            self.cumulative_sizes.append(total)
        print(f"Found {total} examples")

    def __len__(self):
        """Return the total number of examples across all shards."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        """
        Fetch a single (inputs, outputs) pair by global index.

        Maps the global index to the correct shard, parses the TFRecord,
        selects the channels matching ``self.variables``, converts to
        PyTorch float tensors, replaces NaNs with zeros, and optionally
        moves to GPU.

        Parameters
        ----------
        idx : int
            Global sample index in ``[0, len(self))``.

        Returns
        -------
        inputs : torch.Tensor
            Shape ``(C_in, 128, 128)`` where ``C_in`` depends on
            ``variables`` and ``use_ocean``.
        outputs : torch.Tensor
            Shape ``(C_out, 128, 128)`` where ``C_out`` matches the
            number of emulated variables.
        """
        file_idx = 0
        while (
            file_idx < len(self.cumulative_sizes)
            and idx >= self.cumulative_sizes[file_idx]
        ):
            file_idx += 1

        local_idx = idx
        if file_idx > 0:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        dataset = tf.data.TFRecordDataset(self.filenames[file_idx])
        example = next(iter(dataset.skip(local_idx).take(1)))
        parsed = tf.io.parse_single_example(example, self.feature_description)

        if self.variables == ["sit"]:
            inputs, outputs = self._read_tfrecord_sit(parsed)
        elif self.variables == ["sic"]:
            inputs, outputs = self._read_tfrecord_sic(parsed)
        elif self.variables == ["sit", "sic"]:
            inputs, outputs = self._read_tfrecord_sit_sic(parsed)
        elif self.variables == ["siu", "siv"]:
            inputs, outputs = self._read_tfrecord_siu_siv(parsed)
        else:
            inputs, outputs = self._read_tfrecord_all(parsed)

        inputs = torch.from_numpy(inputs.numpy()).float()
        outputs = torch.from_numpy(outputs.numpy()).float()
        inputs = torch.nan_to_num(inputs, 0.0)
        outputs = torch.nan_to_num(outputs, 0.0)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        return inputs, outputs

    def _read_tfrecord_sit(self, parsed):
        """
        Extract sit-only input/output channels from a parsed TFRecord.

        Parameters
        ----------
        parsed : dict
            Output of ``tf.io.parse_single_example``.

        Returns
        -------
        inputs : tf.Tensor
            Shape ``(1 + 7 + N_ocean + N_under, 128, 128)``.
            Channel 0 is sit; remaining channels are atmospheric/ocean forcings.
        outputs : tf.Tensor
            Shape ``(1, 128, 128)``. sit increment.
        """
        inputs = tf.reshape(tf.cast(parsed["inputs"], tf.float32), [self.N_tot_input, 128, 128])
        sit = tf.reshape(inputs[0], [1, 128, 128])
        forcings = inputs[5: 5 + 7 + self.N_ocean + self.N_under]
        inputs = tf.concat([sit, forcings], axis=0)
        outputs = tf.reshape(tf.cast(parsed["outputs"], tf.float32), [5, 128, 128])
        outputs = tf.reshape(outputs[0], [1, 128, 128])
        return inputs, outputs

    def _read_tfrecord_sic(self, parsed):
        """
        Extract sic-only input/output channels from a parsed TFRecord.

        Parameters
        ----------
        parsed : dict
            Output of ``tf.io.parse_single_example``.

        Returns
        -------
        inputs : tf.Tensor
            Shape ``(1 + 7 + N_ocean + N_under, 128, 128)``.
            Channel 0 is sic; remaining channels are atmospheric/ocean forcings.
        outputs : tf.Tensor
            Shape ``(1, 128, 128)``. sic increment.
        """
        inputs = tf.reshape(tf.cast(parsed["inputs"], tf.float32), [self.N_tot_input, 128, 128])
        sic = tf.reshape(inputs[1], [1, 128, 128])
        forcings = inputs[5: 5 + 7 + self.N_ocean + self.N_under]
        inputs = tf.concat([sic, forcings], axis=0)
        outputs = tf.reshape(tf.cast(parsed["outputs"], tf.float32), [5, 128, 128])
        outputs = tf.reshape(outputs[1], [1, 128, 128])
        return inputs, outputs

    def _read_tfrecord_sit_sic(self, parsed):
        """
        Extract sit+sic input/output channels from a parsed TFRecord.

        Parameters
        ----------
        parsed : dict
            Output of ``tf.io.parse_single_example``.

        Returns
        -------
        inputs : tf.Tensor
            Shape ``(2 + 7 + N_ocean + N_under, 128, 128)``.
            Channels 0-1 are sit, sic; remaining channels are forcings.
        outputs : tf.Tensor
            Shape ``(2, 128, 128)``. sit and sic increments.
        """
        inputs = tf.reshape(tf.cast(parsed["inputs"], tf.float32), [self.N_tot_input, 128, 128])
        sit_sic = tf.reshape(inputs[:2], [2, 128, 128])
        forcings = inputs[5: 5 + 7 + self.N_ocean + self.N_under]
        inputs = tf.concat([sit_sic, forcings], axis=0)
        outputs = tf.reshape(tf.cast(parsed["outputs"], tf.float32), [5, 128, 128])
        outputs = tf.reshape(outputs[:2], [2, 128, 128])
        return inputs, outputs

    def _read_tfrecord_siu_siv(self, parsed):
        """
        Extract siu+siv input/output channels from a parsed TFRecord.

        Parameters
        ----------
        parsed : dict
            Output of ``tf.io.parse_single_example``.

        Returns
        -------
        inputs : tf.Tensor
            Shape ``(2 + 7 + N_ocean + N_under, 128, 128)``.
            Channels 0-1 are siu, siv; remaining channels are forcings.
        outputs : tf.Tensor
            Shape ``(2, 128, 128)``. siu and siv increments.
        """
        inputs = tf.reshape(tf.cast(parsed["inputs"], tf.float32), [self.N_tot_input, 128, 128])
        vel = tf.reshape(inputs[2:4], [2, 128, 128])
        forcings = inputs[5: 5 + 7 + self.N_ocean + self.N_under]
        inputs = tf.concat([vel, forcings], axis=0)
        outputs = tf.reshape(tf.cast(parsed["outputs"], tf.float32), [5, 128, 128])
        outputs = tf.reshape(outputs[2:4], [2, 128, 128])
        return inputs, outputs

    def _read_tfrecord_all(self, parsed):
        """
        Return all 5 sea-ice variable channels from a parsed TFRecord.

        Parameters
        ----------
        parsed : dict
            Output of ``tf.io.parse_single_example``.

        Returns
        -------
        inputs : tf.Tensor
            Shape ``(N_tot_input, 128, 128)``. All input channels
            including sea-ice state, forcings, and optional ocean fields.
        outputs : tf.Tensor
            Shape ``(5, 128, 128)``. All 5 variable increments.
        """
        inputs = tf.reshape(tf.cast(parsed["inputs"], tf.float32), [self.N_tot_input, 128, 128])
        outputs = tf.reshape(tf.cast(parsed["outputs"], tf.float32), [5, 128, 128])
        return inputs, outputs

import re
import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config


class Dataset:
    def __init__(self, seed):
        self.seed = seed
        self.conditions = [
            "0.15_Speed_OB",
            "0.275_Speed_OB",
            "0.45_Speed_OB",
            "0.15_Speed_withoutOb",
            "0.275_Speed_withoutOb",
            "0.45_Speed_withoutOb",
        ]
        self.norm = {"imu": {}, "rss": {}, "timestamp": {}}

    def get_npy(self, dir: str, type: str) -> tuple:
        """Loads a .npy file of the given type from the specified directory.

        Args:
            dir (str): Directory containing the .npy files.
            type (str): Type of data to load (e.g., "GT", "IMU", "RssOWP").

        Returns:
            np.ndarray: Loaded data as a NumPy array.
        """
        # Regex check
        pattern = re.compile(rf".*{re.escape(type)}\.npy$")

        # Find file
        matching_file = next((f for f in os.listdir(dir) if pattern.match(f)), None)
        if matching_file is None:
            raise FileNotFoundError(f"No file matching '{type}.npy' found in '{dir}'")

        full_path = os.path.join(dir, matching_file)

        return np.load(full_path)

    def load_data(self, dir: str) -> pd.DataFrame:
        """Loads and concatenates GT, IMU, and RSS data from all condition folders.
        Args:
            dir (str): External Dataset Directory (OWP-IMU).
        Returns:
            dict: A dictionary containing concatenated DataFrames for GT, IMU, and RSS.
        """
        # Create dataset map for obstacle / no obstacle
        gt_list, imu_list, rss_list = [], [], []

        for i, trajFolder in enumerate(self.conditions):
            obs_label = "No Obstacle" if "withoutOb" in trajFolder else "Obstacle"
            path = os.path.join(dir, trajFolder)
            gtNp = self.get_npy(path, "GT")
            gtDf = pd.DataFrame(
                gtNp,
                columns=[
                    "Timestamp",
                    "x",
                    "y",
                    "z",
                    "rot11",
                    "rot12",
                    "rot13",
                    "rot21",
                    "rot22",
                    "rot23",
                    "rot31",
                    "rot32",
                    "rot33",
                ],
            )

            imuNp = self.get_npy(path, "IMU")
            imuDf = pd.DataFrame(imuNp, columns=["Timestamp", "ax", "ay", "az", "gx", "gy", "gz"])

            rssNp = self.get_npy(path, "RssOWP")
            rssDf = pd.DataFrame(rssNp, columns=["Timestamp", "rss1", "rss2", "rss3", "rss4"])

            for df in [gtDf, imuDf, rssDf]:
                df["seq_id"] = i
                df["obstacle"] = obs_label

            gt_list.append(gtDf)
            imu_list.append(imuDf)
            rss_list.append(rssDf)

        # Concatenate all dataframes
        return {
            "gt": pd.concat(gt_list, ignore_index=True),
            "imu": pd.concat(imu_list, ignore_index=True),
            "rss": pd.concat(rss_list, ignore_index=True),
        }

    def preprocess(
        self,
        gt: pd.DataFrame,
        imu: pd.DataFrame,
        rss: pd.DataFrame,
        dir: str,
        data_split: str,
        sample_rate: int,
        window_size: int,
        step_size: int,
        rot=False,
    ) -> dict:
        """
        Preprocesses the dataset by:
            1. Smoothing IMU data with Gaussian filter
            2. Normalizing IMU data to [0, 1]
            3. Power transforming RSS data to [0, 1]
            4. Resampling all modalities to common timestamps
            5. Merging IMU and RSS data into a single input DataFrame
            6. Creating sliding windows with aligned targets

        Args:
            gt (pd.DataFrame): Ground truth DataFrame containing target values.
            imu (pd.DataFrame): IMU DataFrame containing sensor readings.
            rss (pd.DataFrame): RSS DataFrame containing signal strengths.
            dir (str): Directory to save the processed data.
            data_split (str): 'train', 'val' or 'test', determines output folder + normalization.
            sample_rate (int): Highest frequency of inputs (for resampling)
            window_size (int): Size of sliding window
            step_size (int): Step size for sliding window

        Returns:
            dict: A dictionary containing processed input features (X), targets (y), and sequence IDs.
        """

        # Set value columns of each data modality
        imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        rss_cols = ["rss1", "rss2", "rss3", "rss4"]
        gt_cols = ["x", "y", "z"]

        if rot:
            gt_cols += [
                "rot11",
                "rot12",
                "rot13",
                "rot21",
                "rot22",
                "rot23",
                "rot31",
                "rot32",
                "rot33",
            ]

        # Select only needed columns
        gt_df = gt[["Timestamp", "seq_id"] + gt_cols]
        imu_df = imu[["Timestamp", "seq_id"] + imu_cols]
        rss_df = rss[["Timestamp", "seq_id"] + rss_cols]

        # 1. Apply Gaussian filter with sigma = 1
        def smooth_cols(df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
            df = df.copy()
            for col in cols:
                df[col] = gaussian_filter1d(df[col].values, sigma=1)
            return df

        # 2. Normalize sensor data to [0, 1]
        def normalize(df: pd.DataFrame, cols: tuple, key: str, reuse_norm: bool):
            """Normalizes the specified columns of a DataFrame to the range [0, 1].

            Args:
                df (pd.DataFrame): The input DataFrame to normalize.
                cols (list): The list of column names to normalize.
                key (str): 'imu' or 'rss' or 'timestamp'
                reuse_norm (bool): Whether to reuse previous norm params.

            Returns:
                df (pd.DataFrame): The DataFrame with normalized columns.
            """
            df = df.copy()

            if reuse_norm and key in self.norm:
                # Get precomputed min and max
                global_min = pd.Series({col: self.norm[key][col][0] for col in cols})
                global_max = pd.Series({col: self.norm[key][col][1] for col in cols})
            else:
                global_min = df[cols].min()
                global_max = df[cols].max()
                # Store normalization parameters
                self.norm[key] = {col: (global_min[col], global_max[col]) for col in cols}

            df[cols] = (df[cols] - global_min) / (global_max - global_min + 1e-8)
            return df

        # For the log-scaled RSSI values
        def power_transform(df: pd.DataFrame, cols: str, c=-100, beta=2.0) -> pd.DataFrame:
            df = df.copy()
            df[cols] = df[cols].apply(lambda col: ((col - c) / -c) ** beta)
            return df

        reuse_norm = False if data_split == "train" else True
        imu_df = smooth_cols(imu_df, imu_cols)
        imu_df = normalize(imu_df, imu_cols, "imu", reuse_norm)

        rss_df = power_transform(rss_df, rss_cols)
        rss_df = smooth_cols(rss_df, rss_cols)

        # 3. Resample all modalities to common timestamps
        def get_common_timestamps(dfs: list) -> list:
            timestamps = []

            for df in dfs:
                for seq_id, group in df.groupby("seq_id"):
                    # Remove duplicates and sort
                    group = group.drop_duplicates("Timestamp", keep="last").sort_values("Timestamp")
                    timestamps.extend(group["Timestamp"].values)

            # Create common timeline at specified frequency
            if not timestamps:
                return np.array([])

            min_time = min(timestamps)
            max_time = max(timestamps)
            common_timestamps = np.arange(min_time, max_time, 1 / sample_rate)
            return np.unique(common_timestamps)

        # Get common timestamps across all modalities
        common_timestamps = get_common_timestamps([imu_df, rss_df, gt_df])

        def resampling(df: pd.DataFrame, cols: str, common_ts: list) -> pd.DataFrame:
            resampled = []

            for seq_id, group in df.groupby("seq_id"):
                group = group.drop_duplicates("Timestamp", keep="last").sort_values("Timestamp")

                # Create DataFrame with common timestamps
                common_df = pd.DataFrame({"Timestamp": common_ts})
                common_df["seq_id"] = seq_id

                # Interpolate values to common timestamps
                input_vals = group[cols].values
                timestamps = group["Timestamp"].values

                # Skip if no data points to interpolate from
                if len(timestamps) < 2:
                    continue

                interpolated = np.array(
                    [
                        np.interp(common_ts, timestamps, input_vals[:, i])
                        for i in range(input_vals.shape[1])
                    ]
                ).T

                # Add interpolated values to common_df
                for i, col in enumerate(cols):
                    common_df[col] = interpolated[:, i]

                resampled.append(common_df)

            return pd.concat(resampled).reset_index(drop=True) if resampled else pd.DataFrame()

        # Resample all modalities to common timestamps
        imu_resampled = resampling(imu_df, imu_cols, common_timestamps)
        rss_resampled = resampling(rss_df, rss_cols, common_timestamps)
        gt_resampled = resampling(gt_df, gt_cols, common_timestamps)

        # 4. Merge IMU + RSS into 1 input df (now properly aligned)
        merged_df = pd.merge(imu_resampled, rss_resampled, on=["Timestamp", "seq_id"], how="inner")
        # Normalize Timestamps to [0, 1]
        merged_df = normalize(merged_df, ["Timestamp"], "timestamp", reuse_norm)
        merged_df = merged_df.sort_values(["seq_id", "Timestamp"]).reset_index(drop=True)
        gt_resampled = normalize(gt_resampled, ["Timestamp"], "timestamp", reuse_norm)

        # 5. Create sliding windows with aligned targets
        X = []
        y = []
        seq_ids = []

        for seq_id in merged_df["seq_id"].unique():
            # Get inputs and targets for this sequence
            inputs_df = merged_df[merged_df["seq_id"] == seq_id].sort_values("Timestamp")
            targets_df = gt_resampled[gt_resampled["seq_id"] == seq_id].sort_values("Timestamp")

            # Ensure matching timestamps
            common_ts = np.intersect1d(
                inputs_df["Timestamp"].values, targets_df["Timestamp"].values
            )
            if len(common_ts) == 0:
                continue

            inputs_df = inputs_df[inputs_df["Timestamp"].isin(common_ts)]
            targets_df = targets_df[targets_df["Timestamp"].isin(common_ts)]

            inputs = inputs_df[imu_cols + rss_cols].values
            targets = targets_df[gt_cols].values
            timestamps = inputs_df["Timestamp"].values

            # Create windows
            for start in range(0, len(inputs) - window_size + 1, step_size):
                end = start + window_size
                window_inputs = inputs[start:end]
                window_end_time = timestamps[end - 1]

                # Find the target with exactly matching timestamp
                target_idx = np.where(targets_df["Timestamp"].values == window_end_time)[0]

                if len(target_idx) == 0:
                    continue

                window_target = targets[target_idx[0]]

                # Append window details
                X.append(window_inputs)
                y.append(window_target)
                seq_ids.append(seq_id)

        X = np.array(X) if len(X) > 0 else np.empty((0, window_size, len(imu_cols + rss_cols)))
        y = np.array(y) if len(y) > 0 else np.empty((0, len(gt_cols)))
        seq_ids = np.array(seq_ids)

        np.save(os.path.join(dir, f"{data_split}/X.npy"), X)
        np.save(os.path.join(dir, f"{data_split}/y.npy"), y)

        return {"X": X, "y": y, "seq_ids": seq_ids}

    def split_data(self, gt, imu, rss, test_size=0.2, val_size=0.2) -> tuple:
        """Split the dataset into training, validation, and test sets based on sequence IDs.

        Args:
            gt (pd.dataframe): Raw Ground Truth dataframe.
            imu (pd.dataframe): Raw IMU dataframe.
            rss (pd.dataframe): Raw RSS dataframe.
            seq_ids (np.ndarray): Sequence IDs corresponding to each sample.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the dataset to include in the validation split.

        Returns:
            tuple: A tuple containing the training, validation, and test datasets.
        """

        def seq_filter(df, seqs):
            return df[df["seq_id"].isin(seqs)]

        unique_seq_ids = gt["seq_id"].unique()

        train_seq, test_seq = train_test_split(
            unique_seq_ids,
            test_size=test_size,
            random_state=self.seed,
        )
        train_seq, val_seq = train_test_split(
            train_seq,
            test_size=val_size / (1.0 - test_size),
            random_state=self.seed,
        )

        return (
            (seq_filter(gt, train_seq), seq_filter(imu, train_seq), seq_filter(rss, train_seq)),
            (seq_filter(gt, val_seq), seq_filter(imu, val_seq), seq_filter(rss, val_seq)),
            (seq_filter(gt, test_seq), seq_filter(imu, test_seq), seq_filter(rss, test_seq)),
        )

    def create_tf_dataset(self, dir: str, batch_size=32, shuffle=True) -> tf.data.Dataset:
        """Create TensorFlow dataset from .npy files in a directory.

        Args:
            dir (str): Path to the directory containing 'X.npy' and 'y.npy'.
            shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
            batch_size (int, optional): Batch size for the dataset. Defaults to 32.

        Returns:
            tf.data.Dataset: The TensorFlow datasets for training and testing.
        """
        X = np.load(os.path.join(dir, "X.npy"))
        y = np.load(os.path.join(dir, "y.npy"))
        print(X.shape, y.shape)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X), seed=self.seed)

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    config = Config()
    dataset = Dataset(config.random_seed)

    # Load raw data
    raw_data = dataset.load_data(config.external_dataset_dir)

    # Split raw data
    (gt_train, imu_train, rss_train), (gt_val, imu_val, rss_val), (gt_test, imu_test, rss_test) = (
        dataset.split_data(
            raw_data["gt"], raw_data["imu"], raw_data["rss"], config.test_size, config.val_size
        )
    )

    # Process and save each split
    dataset.preprocess(
        gt_train,
        imu_train,
        rss_train,
        config.processed_dataset_dir,
        "train",
        config.sample_rate,
        config.window_size,
        config.step_size,
    )
    dataset.preprocess(
        gt_train,
        imu_train,
        rss_train,
        config.processed_dataset_dir,
        "val",
        config.sample_rate,
        config.window_size,
        config.step_size,
    )
    dataset.preprocess(
        gt_train,
        imu_train,
        rss_train,
        config.processed_dataset_dir,
        "test",
        config.sample_rate,
        config.window_size,
        config.step_size,
    )

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

    def load_data(self, dir: str) -> dict:
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
        
    def load_seperate_data(self, dir: str) -> dict:
        """Loads and concatenates GT, IMU, and RSS data from all condition folders.
        Args:
            dir (str): External Dataset Directory (OWP-IMU).
        Returns:
            dict: A dictionary containing concatenated DataFrames for GT, IMU, and RSS.
        """
        # Create dataset map for obstacle / no obstacle
        condition_dfs = {}

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

            condition_dfs[trajFolder] = {
                "gt": gtDf,
                "imu": imuDf,
                "rss": rssDf
            }

        # Return all dataframes
        return condition_dfs
        
    def split_processing(
        self,
        X: list[pd.DataFrame],
        y: list[pd.DataFrame],
        seq_ids: list[int],
        dir: str,
        data_split: str,
        window_size: int,
        imu_cols: list,
        rss_cols: list
    ):
        """
        Preprocesses the split by:
            1. Smoothing IMU data with Gaussian filter
            2. Normalizing IMU data to [0, 1]
            3. Power transforming RSS data to [0, 1]

        Args:
            X (list[pd.DataFrame]): List of input feature DataFrames.
            y (list[pd.DataFrame]): List of target DataFrames.
            seq_ids (list[int]): List of sequence IDs.
            dir (str): Directory to save the processed data.
            data_split (str): The split type (e.g., "train", "val", "test").
            window_size (int): Size of sliding window.
            imu_cols (list): List of IMU column names.
            rss_cols (list): List of RSS column names.

        Returns:
            dict: A dictionary containing processed input features (X), targets (y), and sequence IDs.
        """

        def add_jitter(df: pd.DataFrame, imu_cols: list[str], rss_cols: list[str], imu_std_scale: float = 0.01, max_jitter: float = 1.0) -> pd.DataFrame:
            """Adds random jitter to specified columns of a DataFrame.

            Args:
                df (pd.DataFrame): The input DataFrame.
                cols (list): List of column names to add jitter to.
                std (float): Standard deviation of the jitter.

            Returns:
                pd.DataFrame: The DataFrame with jitter added to specified columns.
            """
            df = df.copy()
            for col in cols:
                noise = np.random.normal(0, std, size=df[cols].shape)
                df[col] += np.clip(noise, -max_jitter, max_jitter)
            return df
        
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
        
        X_concat = pd.concat(X, ignore_index=True)
        X_concat = add_jitter(X_concat, imu_cols, rss_cols, std=0.01, max_jitter=0.1)
        X_concat = smooth_cols(X_concat, imu_cols)
        X_concat = normalize(X_concat, imu_cols, "imu", reuse_norm)
        X_concat = power_transform(X_concat, rss_cols)
        X_concat = smooth_cols(X_concat, rss_cols)

        X_norm = []
        num_windows = len(X)
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_df = X_concat.iloc[start:end]
            X_norm.append(window_df.values)
        
        X_norm = np.array(X_norm) if len(X_norm) > 0 else np.empty((0, window_size, len(imu_cols + rss_cols)))
        
        seq_ids = np.array(seq_ids)
        
        os.makedirs(dir, exist_ok=True)
        os.makedirs(os.path.join(dir, data_split), exist_ok=True)
        np.save(os.path.join(dir, f"{data_split}/X.npy"), X_norm)
        np.save(os.path.join(dir, f"{data_split}/y.npy"), np.array(y))
        np.save(os.path.join(dir, f"{data_split}/seq_ids.npy"), seq_ids)
        
        print(X_norm.shape)

    def preprocess(self, gt: pd.DataFrame, imu: pd.DataFrame, rss: pd.DataFrame, dir: str, window_size: int, step_size: int, sample_rate: int, test_size=0.2, val_size=0.2):
        """Process the entire dataset by:
           1. Resampling all modalities to common timestamps
           2. Creating sliding windows with aligned targets
           3. Splitting data into train, validation, and test sets
           4. Normalization and Smoothing of IMU and RSS data
           5. Saving processed data to .npy files

        Args:
            gt (pd.dataframe): Raw Ground Truth dataframe.
            imu (pd.dataframe): Raw IMU dataframe.
            rss (pd.dataframe): Raw RSS dataframe.
            dir (str): Directory to save the processed data.    
            window_size (int): Size of sliding window
            step_size (int): Step size for sliding window
            sample_rate (int): Highest frequency of inputs (for resampling)
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the dataset to include in the validation split.

        Returns:
            tuple: A tuple containing the training, validation, and test datasets.
        
        Raises:
            ValueError: If the input dataframes do not have the same length.
        """

        imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        rss_cols = ["rss1", "rss2", "rss3", "rss4"]
        gt_cols = ["x", "y", "z"]
        
        gt_df = gt[["Timestamp", "seq_id"] + gt_cols]
        imu_df = imu[["Timestamp", "seq_id"] + imu_cols]
        rss_df = rss[["Timestamp", "seq_id"] + rss_cols]
        
        # 1. Resample all modalities to common timestamps
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

        imu_resampled = resampling(imu_df, imu_cols, common_timestamps)
        rss_resampled = resampling(rss_df, rss_cols, common_timestamps)
        gt_resampled = resampling(gt_df, gt_cols, common_timestamps)

        # 2. Merge IMU + RSS into 1 input df (now properly aligned)
        merged_df = pd.merge(imu_resampled, rss_resampled, on=["Timestamp", "seq_id"], how="inner")

        # 3. Create sliding windows with aligned targets
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

            timestamps = inputs_df["Timestamp"].values

            # Create windows
            for start in range(0, len(inputs_df) - window_size + 1, step_size):
                end = start + window_size
                window_inputs = inputs_df.iloc[start:end][imu_cols + rss_cols]
                window_end_time = timestamps[end - 1]

                # Find the target with exactly matching timestamp
                target_idx = np.where(targets_df["Timestamp"].values == window_end_time)[0]

                if len(target_idx) == 0:
                    continue

                window_target = targets_df[gt_cols].values[target_idx[0]]

                # Append window details
                X.append(window_inputs)
                y.append(window_target)
                seq_ids.append(seq_id)
        
        # Split into train, validation, and test sets
        
        indices = np.arange(len(y))
        
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.seed,
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size / (1.0 - test_size),
            random_state=self.seed,
        )

        y = np.array(y)
        seq_ids = np.array(seq_ids)
        splits = {
            "train": [[X[i] for i in train_idx], y[train_idx], seq_ids[train_idx]],
            "val": [[X[i] for i in val_idx], y[val_idx], seq_ids[val_idx]],
            "test": [[X[i] for i in test_idx], y[test_idx], seq_ids[test_idx]],
        }
        
        for i in splits:
            sX, sy, sseq_ids = splits[i]
            self.split_processing(sX, sy, sseq_ids, dir, i, window_size, imu_cols, rss_cols)
        
        

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
    
    # # Load raw data together (remember to change step_size = window_size in config)
    # raw_data = dataset.load_data(config.external_dataset_dir)

    # # Process and save each split
    # dataset.preprocess(
    #     raw_data["gt"],
    #     raw_data["imu"],
    #     raw_data["rss"],
    #     config.processed_dataset_dir,
    #     config.window_size,
    #     config.step_size,
    #     config.sample_rate,
    #     config.test_size,
    #     config.val_size,
    # )
    
    # Load separate data 
    raw_data = dataset.load_seperate_data(config.external_dataset_dir)

    # Process and save each split
    for condition, data in raw_data.items():
        dataset.preprocess(
            data["gt"],
            data["imu"],
            data["rss"],
            os.path.join(config.processed_dataset_dir, f"{condition}/"),
            config.window_size,
            config.step_size,
            config.sample_rate,
            config.test_size,
            config.val_size,
        )
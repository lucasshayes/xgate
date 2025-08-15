from itertools import combinations
import pandas as pd
import numpy as np
import os 
import sys
import json
import tensorflow as tf
from typing import TypedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from utils.set_seed import set_seeds

# Typing for windowed data
class WindowedData(TypedDict):
    X: list[pd.DataFrame]
    y: np.ndarray
    seqnos: np.ndarray

class Dataset():
    """Dataset class for handling data loading and preprocessing.
    
    Args:
        seed (int): Random seed for reproducibility.
        target (str): Name of the target column.
        window_size (int, optional): Size of the sliding window. Defaults to 200.
        step_size (int, optional): Step size for the sliding window. Defaults to 100.
    
    Attributes:
        seed (int): Random seed for reproducibility.
        acc_samples (list): List of accelerometer sample columns.
        gateways (list): List of gateway columns.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
        target_classes (list): List of target classes.
        target_col (str): Name of the target column.
        feature_cols (list): List of feature columns.
        norm (dict): Dictionary to hold normalization parameters.
        p_transformer (PowerTransformer): PowerTransformer instance for feature scaling.
        ohe_encoder (OneHotEncoder): OneHotEncoder instance for categorical encoding.
    """
    def __init__(self, seed, target='true_room', window_size=200, step_size=100):
        self.seed = seed
        self.acc_samples = [f's{i}{axis}' for i in range(1, 6) for axis in ['x', 'y', 'z']]
        self.gateways = ['bedroom', 'kitchen', 'living', 'stairs']
        self.target_classes = []
        self.acc_cols = ['ax', 'ay', 'az']
        self.target_col = target
        self.feature_cols = self.acc_cols + self.gateways
        
        self.norm = {}
        self.window_size = window_size
        self.step_size = step_size
        self.p_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        self.ohe_encoder = OneHotEncoder()

    def create_leave_users_out_splits(self, dir: str, val_trajectories: int = 3) -> list[tuple[list[str], list[str]]]:
        files = [f for f in os.listdir(dir) if f.endswith('.csv')]
        user_files = {}
        
        for f in files:
            user = int(f.split('-')[0])
            if user not in user_files:
                user_files[user] = []
            user_files[user].append(f)

        # Create train/val splits for each user
        splits = []
        all_files = [f for files_list in user_files.values() for f in files_list]
        used_files = set()
        
        # Create combinations of exactly val_trajectories files
        for val_files in combinations(all_files, val_trajectories):
            # Get users from these validation files
            val_users = {int(f.split('-')[0]) for f in val_files}
            
            # Skip if any of these files have already been used for validation
            if set(val_files) & used_files:
                continue
            
            # Check if all files from these users are in validation
            # (to maintain user separation - all trajectories from a user go together)
            all_user_files = set()
            for user in val_users:
                all_user_files.update(user_files[user])
            
            # Only proceed if we're taking ALL files from these users
            if set(val_files) == all_user_files:
                train_files = [f for f in all_files if f not in val_files]
                splits.append((train_files, list(val_files)))
                used_files.update(val_files)
                
        return splits

    def load_raw_data(self, dir: str) -> list[pd.DataFrame]:
        """Loads raw data from the specified directory.

        Args:
            dir (str): Directory containing the raw data files.

        Returns:
            list[pd.DataFrame]: List of DataFrames containing the raw data.
        """
        
        files = [f for f in os.listdir(dir) if f.endswith('.csv')]
        data = []
        for file in files:
            df = pd.read_csv(os.path.join(dir, file))
            data.append(df)
        return data
    
    def restructure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restructures the DataFrame for model input.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Restructured DataFrame.
        """
        df = df.copy()

        # Pivot the RSSI values for each gateway, fill na with -105 (~min RSSI)
        rssi = df.pivot_table(index='seqno', columns='gateway', values='rssi', aggfunc='first')
        # Group by 'seqno' and flatten the accelerometer, true_room and timestamp cols
        acc = df.groupby('seqno')[self.acc_samples + ['true_room', 'timestamp']].first()
        # Join the dataset back together
        final_df = acc.join(rssi).reset_index()

        return final_df

    def fit_transforms(self, X_train: list[pd.DataFrame], y_train: np.ndarray):
        """Fits the normalization transformer to the DataFrame.

        Args:
            X_train (list[pd.DataFrame]): The input DataFrames.
            y_train (np.ndarray): The target values for training.
        """
        df = pd.concat(X_train, ignore_index=True)
        
        # Fit the PowerTransformer to the RSSI values
        self.p_transformer.fit(df[self.gateways])
        
        transformed = df.copy()
        transformed[self.gateways] = self.p_transformer.transform(transformed[self.gateways])

        # Fit the OneHotEncoder to the target column
        self.ohe_encoder.fit(y_train.reshape(-1, 1))

        global_min = transformed[self.gateways].min()
        global_max = transformed[self.gateways].max()
        self.norm['rss'] = {col: (global_min[col], global_max[col]) for col in self.gateways}

        imu_min = transformed[self.acc_cols].min()
        imu_max = transformed[self.acc_cols].max()
        self.norm['imu'] = {col: (imu_min[col], imu_max[col]) for col in self.acc_cols}

    def data_pipeline(self, data_splits: dict[str, pd.DataFrame], fit_transforms: bool = True) -> dict[str, WindowedData]:
        """Processes the data splits through the entire pipeline.

        Args:
            data_splits (dict[str, pd.DataFrame]): The data splits to process.
            fit_transforms (bool, optional): Whether to fit the transformations. Defaults to True.

        Returns:
            dict[str, WindowedData]: The processed data splits.
        """
        # Step 1: Restructure data
        splits = {key: self.restructure_data(split) for key, split in data_splits.items()}
        
        # Step 2: Expand accelerometer data
        splits = {key: self.expand_acc(split) for key, split in splits.items()}
        
        # Step 3: Create sliding windows
        splits = {key: self.create_sliding_windows(split, s_cols=["seqno", "sample"]) for key, split in splits.items()}
        
        # Step 4: Fit transforms if requested (usually on training data)
        if fit_transforms and 'train' in splits:
            self.fit_transforms(splits['train']['X'], splits['train']['y'])
        
        # Step 5: Preprocess windows
        splits = {
            key: {**split, 'X': self.preprocess_windows(split['X'], data_split=key)}
            for key, split in splits.items()
        }
        
        return splits
    
    def expand_acc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expands the 5 accelerometer samples into separate rows, interpolating RSSI values based on timestamps.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with expanded accelerometer samples.
        """
        axes = ['x', 'y', 'z']
        
        # Slice arrays for setup 
        rssi_now = df[self.gateways].iloc[:-1].to_numpy()

        # Align targets to next row (end of window later)
        seqnos = df['seqno'].iloc[:-1].to_numpy()
        targets = df[self.target_col].iloc[:-1].to_numpy()
        
        # # Linearly interpolate RSSI values between samples
        rssi_expanded = np.repeat(rssi_now[:, None, :], 5, axis=1)
        
        # Reshape accelerometer data
        acc_raw = df[self.acc_samples].to_numpy().reshape(-1, 5 , 3)
        # Keep all but last row
        acc_df = acc_raw[:-1]
        
        # Create final dataframe with interpolated RSSI values
        out = pd.DataFrame({
            'seqno': np.repeat(seqnos, 5),
            'sample': np.tile(np.arange(1, 6), len(df) - 1),
            self.target_col: np.repeat(targets, 5),
            **{gateway: rssi_expanded[:, :, i].flatten() for i, gateway in enumerate(self.gateways)},
            **{f'a{axis}': acc_df[:, :, j].flatten() for j, axis in enumerate(axes)}
        })

        for gateway in self.gateways:
            # Interpolate NaN values in RSSI columns
            out[gateway] = out[gateway].interpolate(method='linear', limit_direction='both')
            # Forward/backward fill NaN values
            out[gateway] = out[gateway].ffill().bfill()
        
        return out
    
    def create_sliding_windows(self, df: pd.DataFrame, s_cols: list[str]) -> WindowedData:
        """Creates sliding windows from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            s_cols (list[str]): The list of cols to sort by.
            self.window_size (int, optional): Size of the sliding window. Defaults to 200.
            step_size (int, optional): Step size for the sliding window. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - 'X': The DataFrame with sliding windows.
                - 'y': The target values for each window.
                - 'seqnos': The sequence numbers for each window.
        """
        df = df.copy()
        df = df.sort_values(s_cols).reset_index(drop=True)
        X, y, seqnos = [], [], []

        for start in range(0, len(df) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window = df.iloc[start:end]
            
            if len(window) < self.window_size:
                continue
            target_values = window[self.target_col].to_numpy()
            if len(np.unique(target_values)) == 1:
                # Save window features, target and identifier
                X.append(window[self.feature_cols].copy())
                # Save target as last value in window
                y.append(window[self.target_col].to_numpy()[-1])
                # Save first sort column value as identifier
                seqnos.append(window[s_cols[0]].to_numpy()[0])

        y = np.array(y)
        seqnos = np.array(seqnos)

        return {'X': X, 'y': y, 'seqnos': seqnos}
               
    def preprocess_windows(self, X:list[pd.DataFrame], data_split: str = "train", smooth: bool = False) -> list[pd.DataFrame]:
        """Preprocesses the windows by normalizing, transforming, and smoothing the data.

        Args:
            df (list[pd.DataFrame]): The input DataFrames to preprocess.
            data_split (str): Indicates whether the data is for training or testing.
            smooth (bool, optional): Whether to apply smoothing to the data. Defaults to False.

        Returns:
            list[pd.DataFrame]: The list of preprocessed windows.
        """
        X_processed = []
        for df in X:
            df = df.copy()
            # Add noise for acc + rssi
            # if data_split != "test":
            #     df = self.rssi_shift(df, self.gateways, 1)

            # Power transform RSS columns
            df = self.power_transform(df, self.gateways)
            # Normalize IMU + RSS columns
            df = self.normalize(df, self.acc_cols, "imu")
            # df = self.normalize(df, self.gateways, "rss")

            # Option to perform ewma smoothing if needed
            if smooth:
                df = self.smooth_ewma(df, self.acc_cols)
                df = self.smooth_ewma(df, self.gateways)
            X_processed.append(df)

        return X_processed

    def augment(self, X: list[pd.DataFrame], y: np.ndarray, data_split: str = "train", factor: int = 1) -> tuple[list[pd.DataFrame], np.ndarray]:
        """Augments the dataset by applying various transformations to each window.

        Args:
            X (list[pd.DataFrame]): The input DataFrames to augment.
            y (np.ndarray): The target values corresponding to each window.
            factor (int, optional): The augmentation factor (number of times to augment each window). Defaults to 1.

        Returns:
            tuple: A tuple containing the augmented feature DataFrames and target values.
        
        Raises:
            ValueError: If the input data is invalid.
        """
        
        if X is None or len(X) == 0:
            raise ValueError("Input X is empty or None.")
        
        if y is None or len(y) == 0:
            raise ValueError("Input y is empty or None.")

        if len(X) != len(y):
            raise ValueError("Input X and y must have the same length.")

        if factor < 1:
            raise ValueError("Augmentation factor must be at least 1.")

        if data_split != "train":
            return X, y
        
        aug_X = []
        aug_Y = []

        for i, window_X in enumerate(X):
            aug_X.append(window_X.copy())
            aug_Y.append(y[i])

            for j in range(factor):
                aug_window = window_X.copy()
                # Rotate accelerometer data
                aug_window = self.acc_rotation(aug_window, rot_range=10)

                # Add noise to RSSI data
                aug_window = self.add_noise(aug_window, self.gateways, std=1, clip=3)
                aug_window = self.add_noise(aug_window, self.acc_cols, std=0.03, clip=0.1)

                # Add RSSI shift
                aug_window = self.rssi_shift(aug_window, self.gateways, 2)

                aug_X.append(aug_window)
                aug_Y.append(y[i])

        return aug_X, np.array(aug_Y)

    def normalize(self, df: pd.DataFrame, cols: list[str], key: str, reuse_norm: bool = True) -> pd.DataFrame:
        """Normalizes the specified columns of a DataFrame to the range [0, 1].

        Args:
            df (pd.DataFrame): The input DataFrame to normalize.
            cols (list): The list of column names to normalize.
            key (str): 'imu' or 'rss' or 'timestamp'
            reuse_norm (bool, optional): Whether to reuse previous norm params. Defaults to True.

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
        
        range_vals = global_max - global_min
        range_vals = range_vals.where(range_vals > 1e-8, 1e-8) # Avoid division by zero
        df[cols] = (df[cols] - global_min) / range_vals 
        return df

    def power_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Applies a power transformation to specified columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            cols (list): The list of column names to transform.
            reuse_norm (bool, optional): Whether to reuse previous normalization parameters. Defaults to True.
            
        Returns:
            pd.DataFrame: The DataFrame with transformed columns.
        """
        df = df.copy()
        df[cols] = self.p_transformer.transform(df[cols])
        return df

    def add_noise(self, df: pd.DataFrame, cols: list[str], std: float = 0.01, clip: float = 0.1) -> pd.DataFrame:
        """Adds Gaussian noise to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            cols (list[str]): The list of column names to which noise will be added.
            std (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.01.
            clip (float, optional): Maximum jitter value. Defaults to 0.1.

        Returns:
            pd.DataFrame: The DataFrame with added noise.
        """
        df = df.copy()
        for col in cols:
            noise = np.random.normal(0, std, size=df[col].shape)
            df[col] += noise.clip(-clip, clip)
        return df

    def rssi_shift(self, df: pd.DataFrame, cols: list[str], std: int = 4) -> pd.DataFrame:
        """Applies a random shift to the RSSI values in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            cols (list[str]): The list of column names to shift.
            std (int, optional): The standard deviation of the shift. Defaults to 4.

        Returns:
            pd.DataFrame: The DataFrame with shifted RSSI values.
        """
        df = df.copy()
        shift = np.random.normal(0, std, size=(len(df), len(cols)))
        df[cols] += shift
        return df

    def save_data(self, X: list[pd.DataFrame], y: np.ndarray, seqnos: np.ndarray, data_split: str, dir: str):
        """Saves the processed data to a specified directory.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (np.ndarray): The target values.
            seqnos (np.ndarray): The sequence numbers.
            data_split (str): The split of the data (e.g., 'train', 'val', 'test').
            dir (str): The directory to save the data.
        """
        X_array = np.stack([df.to_numpy() for df in X])
        y = self.ohe_encoder.transform(y.reshape(-1, 1)).toarray()

        assert not np.isnan(X_array).any(), "NaNs found in input"
        assert np.all(np.isfinite(X_array)), "Inf or NaN found"
        print(f"Saving {data_split} data:")
        # Check X array stats
        print("-- X shape:", X_array.shape)
        print("-- X mean ± std:", X_array.mean(), X_array.std())
        print("-- X min/max:", X_array.min(), X_array.max())

        # Check y values
        print("-- y unique:", np.unique(y))
        assert y.min() >= 0
        
        target_classes = {}
        for i, room_name in enumerate(self.ohe_encoder.categories_[0]):
            ohe = np.zeros(len(self.ohe_encoder.categories_[0]))
            ohe[i] = 1
            target_classes[str(ohe.tolist())] = room_name
        
        os.makedirs(dir, exist_ok=True)
        os.makedirs(os.path.join(dir, data_split), exist_ok=True)
        try:
            np.save(os.path.join(dir, f"{data_split}/X.npy"), X_array)
            np.save(os.path.join(dir, f"{data_split}/y.npy"), np.array(y))
            np.save(os.path.join(dir, f"{data_split}/seq_ids.npy"), seqnos)
            with open(os.path.join(dir, f"{data_split}/target_classes.json"), 'w') as f:
                json.dump(target_classes, f)
            print(f"Data saved successfully in {dir}/{data_split}/")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

    def create_tf_dataset(self, dir: str, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """Creates a TensorFlow dataset from the saved data files.

        Args:
            dir (str): The directory containing the saved data files.
            batch_size (int, optional): The batch size for the dataset. Defaults to 32.

        Returns:
            tf.data.Dataset: The TensorFlow dataset.
        
        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Directory {dir} does not exist.")
        
        X = np.load(os.path.join(dir, "X.npy")).astype(np.float32)
        y = np.load(os.path.join(dir, "y.npy")).astype(np.float32)
        print(f"Loaded data from {dir}: ")
        print("-- X shape:", X.shape)
        print("-- y shape:", y.shape)

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X), seed=self.seed)

        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return dataset
    
if __name__ == "__main__":
    # Example usage
    config = Config()
    set_seeds(config.random_seed)
    dataset = Dataset(config.random_seed, target='true_room', window_size=config.window_size, step_size=config.step_size)
    
    # 1. Fit transforms across all training data
    print("Fitting global transforms...")
    train = dataset.load_raw_data(config.external_dataset_dir + "train/")
    train_final_df = pd.concat(train, ignore_index=True)
    
    # Process all training data to fit global transforms
    global_data = {"train": train_final_df}
    dataset.data_pipeline(global_data, fit_transforms=True)
    
    # Set target classes
    dataset.target_classes = train_final_df['true_room'].unique().tolist()
    print(f"Target classes: {dataset.target_classes}")
    print("Global transforms fitted!")

    # 2. Create cross-validation splits
    cv_splits = dataset.create_leave_users_out_splits(config.external_dataset_dir + "train/", val_trajectories=3)

    for i, (train_files, val_files) in enumerate(cv_splits):
        print(f"Fold {i + 1}:")
        print("-- Train files:", train_files)
        print("-- Val files:", val_files)

        # 3. Process training and validation data for each fold
        train = [pd.read_csv(os.path.join(config.external_dataset_dir + "train/", f)) for f in train_files]
        val = [pd.read_csv(os.path.join(config.external_dataset_dir + "train/", f)) for f in val_files]
        
        train_df = pd.concat(train, ignore_index=True)
        val_df = pd.concat(val, ignore_index=True)

        splits = {"train": train_df, "val": val_df}
        processed_splits = dataset.data_pipeline(splits, fit_transforms=False)

        # 4. Save for later use
        fold_dir = f"fold_{i + 1}"
        for k, v in processed_splits.items():
            dataset.save_data(
                **v,
                data_split=k,
                dir=os.path.join(config.processed_dataset_dir, fold_dir)
            )
        
        print(f"Data saved for fold {i + 1}")
    
    # Process evaluation train/test split
    test = dataset.load_raw_data(config.external_dataset_dir + "test/")[0]
    eval_data = {"train": train_final_df, "test": test}
    final_splits = dataset.data_pipeline(eval_data, fit_transforms=False)
    
    # Save final evaluation data
    for k, v in final_splits.items():
        dataset.save_data(
            **v,
            data_split=k,
            dir=os.path.join(config.processed_dataset_dir, "eval")
        )

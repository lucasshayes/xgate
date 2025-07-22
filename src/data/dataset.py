import pandas as pd
import numpy as np
import os 
import sys
import json
import tensorflow as tf
from typing import TypedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config

class WindowedData(TypedDict):
    X: list[pd.DataFrame]
    y: np.ndarray
    seqnos: np.ndarray

class Dataset():
    def __init__(self, seed, target, window_size=200, step_size=100):
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
        self.label_encoder = LabelEncoder().fit(self.gateways)

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
    
    def load_train_data(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Loads and concatenates training data from a list of DataFrames.

        Args:
            dfs (list[pd.DataFrame]): List of DataFrames containing training data.

        Returns:
            pd.DataFrame: Concatenated DataFrame of training data.
        """
        train_data = pd.concat(dfs, ignore_index=True)
        return train_data
    
    def restructure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Pivot the RSSI values for each gateway, fill na with -115 (~10 below min RSSI)
        rssi = df.pivot_table(index='seqno', columns='gateway', values='rssi', aggfunc='first').fillna(-115)
        # Group by 'seqno' and flatten the accelerometer, true_room and timestamp cols
        acc = df.groupby('seqno')[self.acc_samples + ['true_room', 'timestamp']].first()
        # Join the dataset back together
        final_df = acc.join(rssi).reset_index()
        return final_df

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
        rssi_next = df[self.gateways].iloc[1:].to_numpy()
        seqnos = df['seqno'].iloc[1:].to_numpy()
        # Align targets to next row (end of window later)
        targets = df[self.target_col].iloc[1:].to_numpy()
        
        # Linearly interpolate RSSI values between samples
        alpha = np.linspace(0, 1, 5)[:, None]
        rssi_interp = (1 - alpha) * rssi_now[:, None, :] + alpha * rssi_next[:, None, :]
        
        # Reshape accelerometer data
        acc_raw = df[self.acc_samples].to_numpy().reshape(-1, 5 , 3)
        # Keep all but last row
        acc_df = acc_raw[:-1]
        
        # Create final dataframe with interpolated RSSI values
        out = pd.DataFrame({
            'seqno': np.repeat(seqnos, 5),
            'sample': np.tile(np.arange(1, 6), len(df) - 1),
            self.target_col: np.repeat(targets, 5),
            # **{} is dictionary comprehension for RSSI and acc values
            **{gateway: rssi_interp[:, :, i].flatten() for i, gateway in enumerate(self.gateways)},
            **{f'a{axis}': acc_df[:, :, j].flatten() for j, axis in enumerate(axes)}
        })
        
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
            
            # Save window features, target and identifier
            X.append(window[self.feature_cols].copy())
            # Save target as last value in window
            y.append(window[self.target_col].array[-1])
            # Save first sort column value as identifier
            seqnos.append(window[s_cols[0]].array[0])
        
        y = np.array(y)
        seqnos = np.array(seqnos)

        return {'X': X, 'y': y, 'seqnos': seqnos}
    
    def split_dataset(self, X: list[pd.DataFrame], y: np.ndarray, seqnos: np.ndarray, test_size: float) -> tuple[WindowedData, WindowedData]:
        """Splits the test dataset into validation and test sets.

        Args:
            X (list[pd.DataFrame]): Windowed feature dataframe
            y (np.ndarray): The target values.
            seqnos (np.ndarray): The sequence numbers.
            test_size (float): Proportion of the dataset to include in the test set.

        Returns:
            tuple[dict, dict]: Validation and test sets.
        """
        idxs = np.arange(len(y))
        
        train_idx, test_idx = train_test_split(
            idxs,
            test_size=test_size,
            random_state=self.seed,
        )

        return ({'X': [X[i] for i in train_idx],'y': y[train_idx], 'seqnos': seqnos[train_idx]},
                {'X': [X[i] for i in test_idx], 'y': y[test_idx], 'seqnos': seqnos[test_idx]}) 
               
    def preprocess(self, X:list[pd.DataFrame], data_split: str = "train", smooth: bool = False) -> list[pd.DataFrame]:
        """Preprocesses the DataFrame by normalizing, transforming, and smoothing the data.

        Args:
            df (list[pd.DataFrame]): The input DataFrames to preprocess.
            data_split (str): Indicates whether the data is for training or testing.
            smooth (bool, optional): Whether to apply smoothing to the data. Defaults to False.

        Returns:
            list[pd.DataFrame]: The list of preprocessed DataFrames.
        """
        df = pd.concat(X, ignore_index=True)
        # Add noise for acc + rssi
        if data_split == "train":
            df = self.add_noise(df, self.acc_cols, std=0.02, clip=0.05)
            df = self.add_noise(df, self.gateways, std=0.005, clip=0.02)
        # Power transform RSS columns
        df = self.power_transform(df, self.gateways, reuse_norm=(data_split != "train"))
        # Normalize IMU + RSS columns
        df = self.normalize(df, self.acc_cols, "imu", reuse_norm=(data_split != "train"))
        df = self.normalize(df, self.gateways, "rss", reuse_norm=(data_split != "train"))

        # Option to perform ewma smoothing if needed
        if smooth:
            df = self.smooth_ewma(df, self.acc_cols)
            df = self.smooth_ewma(df, self.gateways)
        
        X_windowed = []
        num_windows = len(X)
        for i in range(num_windows):
            start = i * self.window_size
            end = start + self.window_size
            window_df = df.iloc[start:end]
            X_windowed.append(window_df)

        return X_windowed

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

        df[cols] = (df[cols] - global_min) / (global_max - global_min + 1e-8)
        return df

    def power_transform(self, df: pd.DataFrame, cols: list[str], reuse_norm: bool = True) -> pd.DataFrame:
        """Applies a power transformation to specified columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            cols (list): The list of column names to transform.
            reuse_norm (bool, optional): Whether to reuse previous normalization parameters. Defaults to True.
            
        Returns:
            pd.DataFrame: The DataFrame with transformed columns.
        """
        df = df.copy()
        df[cols] = self.p_transformer.fit_transform(df[cols]) if not reuse_norm else self.p_transformer.transform(df[cols])
        return df
    
    def smooth_ewma(self, df: pd.DataFrame, cols: list, alpha: float = 0.1) -> pd.DataFrame:
        """Applies Exponential Weighted Moving Average (EWMA) smoothing to specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            cols (list): The list of column names to smooth.
            alpha (float, optional): The smoothing factor. Defaults to 0.1.

        Returns:
            pd.DataFrame: The DataFrame with smoothed columns.
        """
        df = df.copy()
        for col in cols:
            df[col] = df[col].ewm(alpha=alpha, adjust=False).mean()
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
            df[col] += noise
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
        y = np.array(self.label_encoder.transform(y))

        target_classes = {str(i): self.label_encoder.inverse_transform([i])[0] for i in range(len(self.target_classes))}

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
        
        X = np.load(os.path.join(dir, "X.npy"))
        y = np.load(os.path.join(dir, "y.npy"))
        print(np.unique(y))
        print(y.dtype)
        seqnos = np.load(os.path.join(dir, "seq_ids.npy"))

        dataset = tf.data.Dataset.from_tensor_slices((X, y, seqnos))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X), seed=self.seed)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
if __name__ == "__main__":
    # Example usage
    config = Config()
    dataset = Dataset(config.random_seed, target='true_room')
    train = dataset.load_raw_data(config.external_dataset_dir + "train/")
    test = dataset.load_raw_data(config.external_dataset_dir + "test/")[0]
    train = dataset.load_train_data(train)

    dataset.target_classes = train['true_room'].unique().tolist()
    dataset.label_encoder = dataset.label_encoder.fit(dataset.target_classes)
    print(f"True Rooms: {dataset.target_classes}")
    
    splits = {"train": train, "test": test}
    splits = {key: dataset.restructure_data(split) for key, split in splits.items()}
    splits = {key: dataset.expand_acc(split) for key, split in splits.items()}

    splits = {key: dataset.create_sliding_windows(split, s_cols=["seqno", "sample"]) for key, split in splits.items()}
    splits['val'], splits['test'] = dataset.split_dataset(splits['test']['X'], splits['test']['y'], splits['test']['seqnos'], test_size=0.5)
    
    splits = {
        key: {**split, 'X': dataset.preprocess(split['X'], data_split=key)}
              for key, split in splits.items()
    }
    
    print(splits['train']['X'][0].shape)
    print(splits['train']['X'][0].describe())
    
    for k, v in splits.items():
        print(f"--- {k} ---")
        dataset.save_data(
            **v,
            data_split=k,
            dir=config.processed_dataset_dir
        )
    

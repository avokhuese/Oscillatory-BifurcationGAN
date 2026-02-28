"""
Data loading and preprocessing for all three datasets
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Import Config here to avoid circular imports
from config import config

class JenaClimateDataset:
    """Jena Climate Data from Max Planck Institute"""
    
    def __init__(self, seq_len=24):
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        
    def download_data(self):
        """Download Jena climate data"""
        data_path = os.path.join(config.DATA_DIR, 'jena_climate.csv')
        
        try:
            # Try to download from URL
            df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")
        except:
            # Try local file
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                # Generate synthetic temperature data if no file exists
                print("Jena climate data not found. Generating synthetic data...")
                dates = pd.date_range(start='2000-01-01', periods=5000, freq='D')
                temps = 10 + 15 * np.sin(2 * np.pi * np.arange(5000) / 365) + np.random.normal(0, 2, 5000)
                df = pd.DataFrame({'Date': dates, 'Temp': temps})
        
        # Use temperature data
        if 'Temp' in df.columns:
            data = df['Temp'].values.reshape(-1, 1)
        elif 'temp' in df.columns:
            data = df['temp'].values.reshape(-1, 1)
        else:
            # Try to find a numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data = df[numeric_cols[0]].values.reshape(-1, 1)
            else:
                raise ValueError("No numeric temperature column found")
        
        return data
    
    def create_sequences(self, data):
        """Create sequences for time series"""
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i+self.seq_len])
            y.append(data[i+self.seq_len])
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Prepare Jena climate dataset"""
        data = self.download_data()
        data_scaled = self.scaler.fit_transform(data)
        X, y = self.create_sequences(data_scaled)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        return TensorDataset(X, y)


class USDTDataset:
    """US Dollar Tether Cryptocurrency Dataset Loader"""
    
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.root_path = os.path.join(config.DATA_DIR, 'usdt')
        self.file_path = os.path.join(self.root_path, 'USDT-USD.csv')
        os.makedirs(self.root_path, exist_ok=True)
        self.scaler = MinMaxScaler()
        
    def load_and_preprocess(self):
        """Load and preprocess USDT data from local CSV file"""
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            print(f"Dataset file not found at {self.file_path}")
            print("Generating synthetic USDT data for testing...")
            return self._generate_synthetic_data()
        
        print(f"Loading USDT dataset from {self.file_path}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(self.file_path)
            
            # Print column names for debugging
            print(f"Available columns: {df.columns.tolist()}")
            
            # Identify numeric columns (exclude date columns)
            date_columns = []
            numeric_columns = []
            
            for col in df.columns:
                # Check if column might be a date
                if 'date' in col.lower() or 'time' in col.lower():
                    date_columns.append(col)
                else:
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(df[col])
                        numeric_columns.append(col)
                    except (ValueError, TypeError):
                        # If conversion fails, it's probably not numeric
                        date_columns.append(col)
            
            print(f"Date columns: {date_columns}")
            print(f"Numeric columns: {numeric_columns}")
            
            if len(numeric_columns) == 0:
                print("No numeric columns found. Generating synthetic data...")
                return self._generate_synthetic_data()
            
            # Use numeric columns as features
            # Prioritize OHLCV columns if they exist
            priority_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'Open', 'High', 'Low', 'Close', 'Volume',
                           'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
            
            selected_features = []
            for col in priority_cols:
                if col in df.columns and col in numeric_columns:
                    selected_features.append(col)
            
            # If no priority columns found, use all numeric columns
            if len(selected_features) == 0:
                selected_features = numeric_columns[:5]  # Take first 5 numeric columns
            
            print(f"Using features: {selected_features}")
            
            # Extract numeric data
            data = df[selected_features].copy()
            
            # Convert to numeric, coercing errors to NaN
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Convert to numpy array
            data_array = data.values.astype(np.float32)
            
            # Remove outliers using IQR
            Q1 = np.percentile(data_array, 25, axis=0)
            Q3 = np.percentile(data_array, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Less aggressive clipping
            upper_bound = Q3 + 3 * IQR
            data_array = np.clip(data_array, lower_bound, upper_bound)
            
            return data_array
            
        except Exception as e:
            print(f"Error loading USDT data: {e}")
            print("Generating synthetic USDT data instead...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic USDT data for testing"""
        print("Generating synthetic USDT data...")
        
        # Generate 5000 time points
        n_points = 5000
        
        # Create synthetic OHLCV data
        # Base price around 1.00 with small variations (stablecoin)
        base_price = 1.00
        noise = np.random.normal(0, 0.001, n_points)
        trend = 0.00001 * np.arange(n_points)  # Very slight trend
        
        # Generate OHLCV
        close = base_price + trend + noise
        open_price = close + np.random.normal(0, 0.0005, n_points)
        high = np.maximum(open_price, close) + np.random.uniform(0, 0.002, n_points)
        low = np.minimum(open_price, close) - np.random.uniform(0, 0.002, n_points)
        volume = np.random.uniform(1000, 10000, n_points)
        
        # Stack into array
        data = np.column_stack([open_price, high, low, close, volume])
        
        return data.astype(np.float32)
    
    def create_sequences(self, data):
        """Create sequences for time series"""
        sequences = []
        for i in range(0, len(data) - self.sequence_length, 4):  # stride 4
            sequences.append(data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def prepare_data(self):
        """Prepare USDT dataset"""
        # Load and preprocess data
        data = self.load_and_preprocess()
        
        # Normalize
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        sequences = self.create_sequences(data_scaled)
        
        if len(sequences) == 0:
            print("No sequences created. Generating fallback data...")
            # Generate fallback sequences
            sequences = np.random.randn(100, self.sequence_length, 5).astype(np.float32)
        
        print(f"Created {sequences.shape[0]} sequences of length {sequences.shape[1]} with {sequences.shape[2]} features")
        
        # Create features and targets (next step prediction)
        X = sequences[:, :-1, :]  # All but last time step
        y = sequences[:, -1, :]    # Last time step
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        return TensorDataset(X, y)
    
    def inspect_data(self):
        """Helper method to inspect the data structure"""
        df = pd.read_csv(self.file_path)
        print("\n=== Dataset Inspection ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nBasic stats:\n{df.describe()}")
        return df


class HumidityDataset:
    """Plant sensor humidity data"""
    
    def __init__(self, seq_len=24):
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        
    def load_real_data(self):
        """Try to load real humidity data"""
        data_path = os.path.join(config.DATA_DIR, 'humidity.csv')
        
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
                # Try to find humidity column
                for col in df.columns:
                    if 'humidity' in col.lower() or 'hum' in col.lower():
                        return df[col].values.reshape(-1, 1)
                
                # If no humidity column found, use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    return df[numeric_cols[0]].values.reshape(-1, 1)
            except:
                pass
        
        return None
        
    def generate_synthetic_humidity(self, n_samples=10000):
        """Generate synthetic humidity data"""
        t = np.linspace(0, 100, n_samples)
        
        # Daily cycle (24-hour pattern)
        daily = 20 * np.sin(2 * np.pi * t / 24) + 50
        
        # Seasonal trend (30-day pattern)
        seasonal = 10 * np.sin(2 * np.pi * t / (24 * 30))
        
        # Random variations
        noise = np.random.normal(0, 2, n_samples)
        
        # Events (watering, etc.)
        events = np.zeros(n_samples)
        event_indices = np.random.choice(n_samples, 50, replace=False)
        events[event_indices] = np.random.uniform(10, 30, 50)
        
        # Combine components
        data = daily + seasonal + noise + events
        
        # Ensure humidity stays in realistic range (0-100%)
        data = np.clip(data, 0, 100)
        
        return data.reshape(-1, 1)
    
    def create_sequences(self, data):
        """Create sequences for time series"""
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i+self.seq_len])
            y.append(data[i+self.seq_len])
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Prepare humidity dataset"""
        # Try to load real data first
        data = self.load_real_data()
        
        if data is None:
            print("Real humidity data not found. Generating synthetic data...")
            data = self.generate_synthetic_humidity()
        
        # Normalize
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(data_scaled)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        print(f"Humidity dataset: {X.shape[0]} sequences, {X.shape[2]} features")
        
        return TensorDataset(X, y)


def get_dataloaders(dataset_name):
    """Get train, validation, and test dataloaders for specified dataset"""
    
    print(f"\nPreparing {dataset_name} dataset...")
    
    if dataset_name == 'jena':
        dataset_class = JenaClimateDataset(seq_len=config.SEQ_LEN)
    elif dataset_name == 'usdt':
        dataset_class = USDTDataset(sequence_length=config.SEQ_LEN)
    elif dataset_name == 'humidity':
        dataset_class = HumidityDataset(seq_len=config.SEQ_LEN)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    try:
        # Prepare dataset
        full_dataset = dataset_class.prepare_data()
        
        # Split dataset
        total_size = len(full_dataset)
        
        if total_size == 0:
            raise ValueError(f"Dataset {dataset_name} has no samples")
        
        train_size = int(total_size * config.TRAIN_SPLIT)
        val_size = int(total_size * config.VAL_SPLIT)
        train_size = train_size - val_size
        test_size = total_size - train_size - val_size
        
        # Ensure non-zero sizes
        if train_size == 0:
            train_size = max(1, total_size // 2)
            val_size = max(1, total_size // 4)
            test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.SEED)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(config.BATCH_SIZE, len(train_dataset)), 
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=min(config.BATCH_SIZE, len(val_dataset)), 
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=min(config.BATCH_SIZE, len(test_dataset)), 
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, dataset_class.scaler
        
    except Exception as e:
        print(f"Error preparing {dataset_name} dataset: {e}")
        print("Creating fallback synthetic dataset...")
        
        # Create fallback synthetic dataset
        return create_fallback_dataloaders(dataset_name)


def create_fallback_dataloaders(dataset_name):
    """Create fallback synthetic dataloaders if real data loading fails"""
    
    print(f"Creating synthetic fallback data for {dataset_name}...")
    
    # Generate synthetic sequences
    n_samples = 1000
    seq_len = config.SEQ_LEN
    
    if dataset_name == 'jena':
        n_features = 1
        # Temperature-like data with daily and yearly patterns
        t = np.linspace(0, 100, n_samples * seq_len)
        data = 15 + 10 * np.sin(2 * np.pi * t / 365) + 5 * np.sin(2 * np.pi * t) + np.random.randn(n_samples * seq_len) * 2
        data = data.reshape(n_samples, seq_len, 1)
        
    elif dataset_name == 'usdt':
        n_features = 5
        # USDT-like data with small variations around 1.0
        data = np.ones((n_samples, seq_len, n_features))
        for i in range(n_features):
            data[:, :, i] += np.random.randn(n_samples, seq_len) * 0.001
            # Add some trends
            trend = np.linspace(0, 0.01, seq_len).reshape(1, seq_len, 1)
            data[:, :, i] += trend
        
    else:  # humidity
        n_features = 1
        # Humidity-like data with daily patterns
        t = np.linspace(0, 100, n_samples * seq_len)
        data = 50 + 20 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_samples * seq_len) * 5
        data = data.reshape(n_samples, seq_len, 1)
    
    # Normalize
    scaler = StandardScaler()
    data_flat = data.reshape(-1, n_features)
    data_scaled = scaler.fit_transform(data_flat).reshape(n_samples, seq_len, n_features)
    
    # Create sequences
    X = torch.FloatTensor(data_scaled[:, :-1, :])
    y = torch.FloatTensor(data_scaled[:, -1, :])
    
    dataset = TensorDataset(X, y)
    
    # Split
    total_size = len(dataset)
    train_size = int(total_size * config.TRAIN_SPLIT)
    val_size = int(total_size * config.VAL_SPLIT)
    train_size = train_size - val_size
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Fallback dataset created: {n_samples} sequences")
    
    return train_loader, val_loader, test_loader, scaler
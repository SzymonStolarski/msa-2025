import pandas as pd
import numpy as np


def create_features_and_target_garch_xgb(returns: pd.Series, garch_variances: pd.Series, lags: int = 5,
                               target_col: str = 'squared_returns') -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for XGBoost volatility forecasting with GARCH predictions as an additional feature.
    - Features: Lagged returns for the given number of lags + current GARCH variance (no additional shift).
    - Target: Squared returns as a proxy for variance/volatility.

    Assumes returns and garch_variances are pd.Series with the same date index (garch_variances is h_t for each t).
    Returns X (features DataFrame) and y (target Series), aligned by index.
    Drops rows with NaN values due to lagging.
    """
    data = pd.DataFrame({target_col: returns ** 2})
    
    # Create lagged features based on returns
    for lag in range(1, lags + 1):
        data[f'return_lag_{lag}'] = returns.shift(lag)
    
    # Add current GARCH variance (h_t for predicting r_t^2)
    data['garch_variance'] = garch_variances
    
    # Drop NaN rows
    data = data.dropna()
    
    X = data[[col for col in data.columns if 'lag' in col or 'garch' in col]]
    y = data[target_col]
    
    return X, y

def create_features_and_target_xgb(returns: pd.Series, lags: int = 5, target_col: str = 'squared_returns') -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for XGBoost volatility forecasting.
    - Features: Lagged returns (as specified: opóźnienia zwrotów) for the given number of lags.
    - Target: Squared returns as a proxy for variance/volatility.

    Assumes returns is pd.Series with date index.
    Returns X (features DataFrame) and y (target Series), aligned by index.
    Drops rows with NaN values due to lagging.
    """
    data = pd.DataFrame({target_col: returns ** 2})
    
    # Create lagged features based on returns (not squared)
    for lag in range(1, lags + 1):
        data[f'return_lag_{lag}'] = returns.shift(lag)
    
    # Drop NaN rows
    data = data.dropna()
    
    X = data[[col for col in data.columns if 'lag' in col]]
    y = data[target_col]
    
    return X, y

def create_sequences(returns: pd.Series, seq_length: int = 5, target_col: str = 'squared_returns') -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM volatility forecasting.
    - Sequences: Lagged returns shaped as (samples, seq_length, 1) for LSTM input.
    - Target: Squared returns as a proxy for variance/volatility.

    Assumes returns is pd.Series with date index.
    Returns X (3D np.array for LSTM) and y (1D np.array), aligned.
    """
    data = returns.values
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Sequence of returns
        y.append(data[i + seq_length] ** 2)  # Target: squared return at next step
    
    X = np.array(X).reshape(-1, seq_length, 1)  # (samples, seq_length, features=1)
    y = np.array(y)
    
    return X, y

def create_sequences_garch(returns: pd.Series, garch_variances: pd.Series, seq_length: int = 5,
                     target_col: str = 'squared_returns') -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for hybrid LSTM volatility forecasting with GARCH variances.
    - Sequences: Lagged returns (t-seq to t-1) and shifted GARCH variances (t-seq+1 to t) shaped as (samples, seq_length, 2).
    - Target: Squared returns as a proxy for variance/volatility at t.

    This alignment allows the model to use the current GARCH prediction (h_t) in the last time step while using past returns up to t-1.
    Assumes returns and garch_variances are pd.Series with the same date index.
    Returns X (3D np.array for LSTM: samples, seq_length, 2) and y (1D np.array), aligned.
    """
    returns_data = returns.values
    garch_data = garch_variances.values
    X, y = [], []
    for i in range(len(returns_data) - seq_length):
        returns_seq = returns_data[i:i + seq_length]  # returns from i to i+seq-1 (t-seq to t-1)
        garch_seq = garch_data[i + 1:i + seq_length + 1]  # garch from i+1 to i+seq (t-seq+1 to t)
        seq = np.column_stack((returns_seq, garch_seq)).reshape(seq_length, 2)
        X.append(seq)
        y.append(returns_data[i + seq_length] ** 2)  # squared return at i+seq (t)
    
    X = np.array(X)  # (samples, seq_length, 2)
    y = np.array(y)
    
    return X, y

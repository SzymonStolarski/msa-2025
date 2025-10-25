import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.arch import ARCH

from functions.dataprep import create_sequences, create_sequences_garch


def build_lstm_model(seq_length: int = 5, input_features: int = 1, hidden_size: int = 50, num_layers: int = 1, learning_rate: float = 0.001) -> Sequential:
    """
    Build a Keras LSTM model for variance prediction.
    """
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(hidden_size, input_shape=(seq_length, input_features), return_sequences=(num_layers > 1)))
        else:
            model.add(LSTM(hidden_size, return_sequences=(i < num_layers - 1)))
    model.add(Dense(1)) # Output for variance prediction
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def evaluate_lstm(train_returns: pd.Series, test_returns: pd.Series, lstm_params: dict,
                   reestimate_every: int = 20) -> pd.Series:
    """
    Evaluation function for LSTM model for volatility forecasting using TensorFlow/Keras.
    - Builds and fits the model on the initial training set using sequences.
    - For each day in the test set, performs a 1-step forecast of squared returns.
    - Updates the data by appending actual returns.
    - Every 'reestimate_every' days (default 20), re-builds and re-fits the model on the expanding window of data.
    - Uses StandardScaler for input sequences and targets to improve training stability.

    Assumes train_returns and test_returns are pd.Series with date index.
    Returns pd.Series with forecasted squared returns (variance proxy) for the test_returns index.
    """
    seq_length = lstm_params.get('seq_length', 5)
    hidden_size = lstm_params.get('hidden_size', 50)
    num_layers = lstm_params.get('num_layers', 1)
    batch_size = lstm_params.get('batch_size', 32)
    epochs = lstm_params.get('epochs', 50)
    learning_rate = lstm_params.get('learning_rate', 0.001)
    # Copy of training data (expanding window for returns)
    current_returns = train_returns.copy()
    
    # Initial data preparation
    X_train, y_train = create_sequences(current_returns, seq_length=seq_length)
    
    # Scalers: one for inputs (returns), one for targets (squared returns)
    scaler_input = StandardScaler()
    scaler_target = StandardScaler()
    
    # Fit scalers
    # For inputs: fit on flattened returns
    scaler_input.fit(current_returns.values.reshape(-1, 1))
    # For targets: fit on y_train
    scaler_target.fit(y_train.reshape(-1, 1))
    
    # Scale data
    X_train_scaled = np.zeros_like(X_train)
    for i in range(X_train.shape[0]):
        X_train_scaled[i] = scaler_input.transform(X_train[i])
    y_train_scaled = scaler_target.transform(y_train.reshape(-1, 1)).flatten()
    
    # Build and train model
    model = build_lstm_model(seq_length=seq_length, hidden_size=hidden_size, num_layers=num_layers, input_features=1, learning_rate=learning_rate)
    model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # List for predictions
    predictions = []
    
    # Step counter for re-estimation
    step = 0
    
    # Loop over test days
    for date in test_returns.index:
        # Prepare sequence for the current prediction (1-step ahead)
        # Use the last 'seq_length' returns from current_returns
        last_sequence = current_returns.iloc[-seq_length:].values.reshape(1, seq_length, 1)
        
        # Scale the input sequence
        last_sequence_scaled = np.zeros_like(last_sequence)
        for i in range(last_sequence.shape[1]):
            last_sequence_scaled[0, i, 0] = scaler_input.transform([[last_sequence[0, i, 0]]])[0][0]
        
        # 1-step forecast (scaled)
        pred_var_scaled = model.predict(last_sequence_scaled, verbose=0)[0][0]
        
        # Inverse scale the prediction
        pred_var = scaler_target.inverse_transform([[pred_var_scaled]])[0][0]
        pred_var = max(0, pred_var)  # Clip negatives to 0 without flattening positives - freshly added
        predictions.append(pred_var)
        
        # Current actual return
        actual_return = test_returns.loc[date]
        
        # Add the actual return to the current data (expanding window)
        current_returns = pd.concat([current_returns, pd.Series([actual_return], index=[date])])
        
        step += 1
        
        # Re-build and re-fit the model every 'reestimate_every' days on the expanded window
        if step % reestimate_every == 0:
            X_train, y_train = create_sequences(current_returns, seq_length=seq_length)
            
            # Re-fit scalers on updated data
            scaler_input = StandardScaler()
            scaler_input.fit(current_returns.values.reshape(-1, 1))
            scaler_target = StandardScaler()
            scaler_target.fit(y_train.reshape(-1, 1))
            
            # Scale updated data
            X_train_scaled = np.zeros_like(X_train)
            for i in range(X_train.shape[0]):
                X_train_scaled[i] = scaler_input.transform(X_train[i])
            y_train_scaled = scaler_target.transform(y_train.reshape(-1, 1)).flatten()
            
            model = build_lstm_model(seq_length=seq_length, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate, input_features=1)
            model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Return predictions as pd.Series with test index
    return pd.Series(predictions, index=test_returns.index, name='predicted_variance')


def evaluate_hybrid_lstm_with_garch(train_returns: pd.Series, test_returns: pd.Series, garch_params: dict,
                                    lstm_params: dict, reestimate_every: int = 20) -> pd.Series:
    """
    Evaluation function for hybrid LSTM model incorporating GARCH predictions as part of the sequence.
    - Fits GARCH on the initial training set, obtains in-sample variances.
    - Prepares sequences including shifted GARCH variances and fits LSTM.
    - For each day in the test set, computes 1-step GARCH forecast (h_t), builds sequence with past returns (t-seq to t-1) and GARCH (t-seq+1 to t), predicts squared returns at t.
    - Appends actual return and updates states.
    - Every 'reestimate_every' days, re-fits both models on the expanding window.
    - Uses separate StandardScalers for returns, GARCH variances (inputs), and targets.

    Assumes train_returns and test_returns are pd.Series with date index.
    Returns pd.Series with forecasted squared returns (variance proxy) for the test_returns index.
    """
    p = garch_params.get('p', 1)
    o = garch_params.get('o', 0)
    q = garch_params.get('q', 1)
    dist = garch_params.get('dist', "Normal")
    vol = garch_params.get('vol', 'GARCH')
    method = garch_params.get('method', 'analytic')
    random_state = garch_params.get('random_state', 42)
    rescale = garch_params.get('rescale', False)
    # GARCH Forecaster initialization
    garch_forecaster = ARCH(p=p, o=o, q=q, dist=dist, vol=vol, method=method, random_state=random_state, rescale=rescale)

    seq_length = lstm_params.get('seq_length', 5)
    hidden_size = lstm_params.get('hidden_size', 50)
    num_layers = lstm_params.get('num_layers', 1)
    batch_size = lstm_params.get('batch_size', 32)
    epochs = lstm_params.get('epochs', 50)
    learning_rate = lstm_params.get('learning_rate', 0.001)
    
    # Copy of training data (expanding window for returns)
    current_returns = train_returns.copy()
    
    # Initial GARCH fitting and in-sample variances
    garch_forecaster.fit(current_returns)
    results = garch_forecaster._fitted_forecaster
    garch_in_sample = pd.Series(results.conditional_volatility ** 2, index=current_returns.index, name='garch_variance')
    
    # Extract GARCH parameters for manual recursion
    params = results.params
    omega = params['omega']
    alphas = [params.get(f'alpha[{i}]', 0.0) for i in range(1, p + 1)]
    gammas = [params.get(f'gamma[{k}]', 0.0) for k in range(1, o + 1)]
    betas = [params.get(f'beta[{j}]', 0.0) for j in range(1, q + 1)]
    
    # Initial GARCH buffers: lists with oldest to newest
    last_r_sq_list = list(current_returns.iloc[-p:]**2) if p > 0 else []
    last_neg_ind_r_sq_list = [(r**2 if r < 0 else 0.0) for r in current_returns.iloc[-o:]] if o > 0 else []
    last_h_list = list(results.conditional_volatility[-q:]**2) if q > 0 else []
    
    # Pad buffers if necessary
    if len(last_r_sq_list) < p:
        last_r_sq_list = [0.0] * (p - len(last_r_sq_list)) + last_r_sq_list
    if len(last_neg_ind_r_sq_list) < o:
        last_neg_ind_r_sq_list = [0.0] * (o - len(last_neg_ind_r_sq_list)) + last_neg_ind_r_sq_list
    if len(last_h_list) < q:
        last_h_list = [0.0] * (q - len(last_h_list)) + last_h_list
    
    # Initial data preparation for LSTM using in-sample GARCH
    X_train, y_train = create_sequences_garch(current_returns, garch_in_sample, seq_length=seq_length)
    
    # Scalers: separate for returns and garch (inputs), and for targets
    scaler_returns = StandardScaler()
    scaler_garch = StandardScaler()
    scaler_target = StandardScaler()
    
    # Fit scalers
    scaler_returns.fit(current_returns.values.reshape(-1, 1))
    scaler_garch.fit(garch_in_sample.values.reshape(-1, 1))
    scaler_target.fit(y_train.reshape(-1, 1))
    
    # Scale X_train: scale each channel separately
    X_train_scaled = np.zeros_like(X_train)
    X_train_scaled[:, :, 0] = scaler_returns.transform(X_train[:, :, 0].reshape(-1, 1)).reshape(X_train.shape[0], seq_length)
    X_train_scaled[:, :, 1] = scaler_garch.transform(X_train[:, :, 1].reshape(-1, 1)).reshape(X_train.shape[0], seq_length)
    y_train_scaled = scaler_target.transform(y_train.reshape(-1, 1)).flatten()
    
    # Build and train model
    model = build_lstm_model(seq_length=seq_length, input_features=2, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate)
    model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # List for predictions (LSTM out-of-sample)
    predictions = []
    
    # Track current GARCH variances (start with in-sample)
    current_garch_variances = garch_in_sample.copy()
    
    # Step counter for re-estimation
    step = 0
    
    # Loop over test days
    for date in test_returns.index:
        # Compute the current GARCH 1-step prediction (h_t based on t-1 state)
        arch_part = sum(alphas[i-1] * last_r_sq_list[-i] for i in range(1, p + 1)) if p > 0 else 0.0
        asym_part = sum(gammas[k-1] * last_neg_ind_r_sq_list[-k] for k in range(1, o + 1)) if o > 0 else 0.0
        garch_part = sum(betas[j-1] * last_h_list[-j] for j in range(1, q + 1)) if q > 0 else 0.0
        pred_h_garch = omega + arch_part + asym_part + garch_part
        
        # Prepare sequence for the current prediction
        # Returns sequence: last seq_length returns (up to t-1)
        returns_seq = current_returns.iloc[-seq_length:].values.reshape(seq_length, 1)
        
        # GARCH sequence: last (seq_length-1) garch (up to h_{t-1}) + pred_h_garch (h_t)
        if len(current_garch_variances) >= seq_length - 1:
            garch_seq = current_garch_variances.iloc[-(seq_length - 1):].values.tolist() + [pred_h_garch]
        else:
            # Pad if not enough history (early steps)
            pad = [0.0] * (seq_length - 1 - len(current_garch_variances))
            garch_seq = pad + current_garch_variances.values.tolist() + [pred_h_garch]
        garch_seq = np.array(garch_seq).reshape(seq_length, 1)
        
        # Combine into (1, seq_length, 2)
        last_sequence = np.concatenate((returns_seq, garch_seq), axis=1).reshape(1, seq_length, 2)
        
        # Scale the input sequence
        last_sequence_scaled = np.zeros_like(last_sequence)
        last_sequence_scaled[:, :, 0] = scaler_returns.transform(last_sequence[:, :, 0].reshape(-1, 1)).reshape(1, seq_length)
        last_sequence_scaled[:, :, 1] = scaler_garch.transform(last_sequence[:, :, 1].reshape(-1, 1)).reshape(1, seq_length)
        
        # 1-step forecast (scaled)
        pred_var_scaled = model.predict(last_sequence_scaled, verbose=0)[0][0]
        
        # Inverse scale the prediction
        pred_var = scaler_target.inverse_transform([[pred_var_scaled]])[0][0]
        pred_var = max(0, pred_var)  # Clip negatives to 0 without flattening positives - freshly added
        predictions.append(pred_var)
        
        # Append GARCH prediction to current_garch_variances
        current_garch_variances = pd.concat([current_garch_variances, pd.Series([pred_h_garch], index=[date])])
        
        # Current actual return
        actual_return = test_returns.loc[date]
        
        # Update GARCH buffers
        new_r_sq = actual_return ** 2
        new_neg_ind_r_sq = new_r_sq if actual_return < 0 else 0.0
        
        if p > 0:
            last_r_sq_list.append(new_r_sq)
            if len(last_r_sq_list) > p:
                last_r_sq_list.pop(0)
        
        if o > 0:
            last_neg_ind_r_sq_list.append(new_neg_ind_r_sq)
            if len(last_neg_ind_r_sq_list) > o:
                last_neg_ind_r_sq_list.pop(0)
        
        if q > 0:
            last_h_list.append(pred_h_garch)
            if len(last_h_list) > q:
                last_h_list.pop(0)
        
        # Add the actual return to the current data
        current_returns = pd.concat([current_returns, pd.Series([actual_return], index=[date])])
        
        step += 1
        
        # Re-estimate both models every 'reestimate_every' days
        if step % reestimate_every == 0:
            # Re-fit GARCH
            garch_forecaster.fit(current_returns)
            results = garch_forecaster._fitted_forecaster
            params = results.params
            omega = params['omega']
            alphas = [params.get(f'alpha[{i}]', 0.0) for i in range(1, p + 1)]
            gammas = [params.get(f'gamma[{k}]', 0.0) for k in range(1, o + 1)]
            betas = [params.get(f'beta[{j}]', 0.0) for j in range(1, q + 1)]
            
            # Reset GARCH buffers and update in-sample variances
            current_garch_variances = pd.Series(results.conditional_volatility ** 2, index=current_returns.index)
            
            last_r_sq_list = list(current_returns.iloc[-p:]**2) if p > 0 else []
            last_neg_ind_r_sq_list = [(r**2 if r < 0 else 0.0) for r in current_returns.iloc[-o:]] if o > 0 else []
            last_h_list = list(results.conditional_volatility[-q:]**2) if q > 0 else []
            
            # Pad if necessary
            if len(last_r_sq_list) < p:
                last_r_sq_list = [0.0] * (p - len(last_r_sq_list)) + last_r_sq_list
            if len(last_neg_ind_r_sq_list) < o:
                last_neg_ind_r_sq_list = [0.0] * (o - len(last_neg_ind_r_sq_list)) + last_neg_ind_r_sq_list
            if len(last_h_list) < q:
                last_h_list = [0.0] * (q - len(last_h_list)) + last_h_list
            
            # Prepare updated sequences for LSTM
            X_train, y_train = create_sequences_garch(current_returns, current_garch_variances, seq_length=seq_length)
            
            # Re-fit scalers on updated data
            scaler_returns = StandardScaler()
            scaler_returns.fit(current_returns.values.reshape(-1, 1))
            scaler_garch = StandardScaler()
            scaler_garch.fit(current_garch_variances.values.reshape(-1, 1))
            scaler_target = StandardScaler()
            scaler_target.fit(y_train.reshape(-1, 1))
            
            # Scale updated X_train
            X_train_scaled = np.zeros_like(X_train)
            X_train_scaled[:, :, 0] = scaler_returns.transform(X_train[:, :, 0].reshape(-1, 1)).reshape(X_train.shape[0], seq_length)
            X_train_scaled[:, :, 1] = scaler_garch.transform(X_train[:, :, 1].reshape(-1, 1)).reshape(X_train.shape[0], seq_length)
            y_train_scaled = scaler_target.transform(y_train.reshape(-1, 1)).flatten()
            
            # Re-build and re-fit LSTM
            model = build_lstm_model(seq_length=seq_length, input_features=2, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate)
            model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Return LSTM predictions as pd.Series with test index
    return pd.Series(predictions, index=test_returns.index, name='predicted_variance_hybrid')

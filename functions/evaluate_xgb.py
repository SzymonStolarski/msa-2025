import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arch import ARCH
from xgboost import XGBRegressor
import numpy as np

from functions.dataprep import create_features_and_target_garch_xgb, create_features_and_target_xgb


def evaluate_xgboost(train_returns: pd.Series, test_returns: pd.Series, xgboost_params:dict, lags: int = 5,
                     reestimate_every: int = 20) -> pd.Series:
    """
    Evaluation function for XGBoost model for volatility forecasting.
    - Fits the model on the initial training set using prepared features and target (squared returns).
    - For each day in the test set, performs a 1-step forecast of squared returns (volatility proxy).
    - Updates the data by appending actual returns.
    - Every 'reestimate_every' days (default 20), re-fits the model on the expanding window of data.

    Assumes train_returns and test_returns are pd.Series with date index.
    Returns pd.Series with forecasted squared returns (variance proxy) for the test_returns index.
    """
    objective = xgboost_params.get('objective', 'reg:squarederror')
    n_estimators = xgboost_params.get('n_estimators', 100)
    learning_rate = xgboost_params.get('learning_rate', 0.1)
    max_depth = xgboost_params.get('max_depth', 3)
    eval_metric = xgboost_params.get('eval_metric', 'rmse')
    random_state = xgboost_params.get('random_state', 42)

    # XGBoost regressor initialization (simple defaults; can tune if needed)
    model = XGBRegressor(objective=objective, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth,
                         eval_metric=eval_metric, random_state=random_state)
    
    # Copy of training data (expanding window for returns)
    current_returns = train_returns.copy()
    
    # Initial data preparation and fitting
    X_train, y_train = create_features_and_target_xgb(current_returns, lags=lags)
    model.fit(X_train, y_train)
    
    # List for predictions
    predictions = []
    
    # Step counter for re-estimation
    step = 0
    
    # Loop over test days
    for date in test_returns.index:
        # Prepare features for the current prediction (1-step ahead)
        # Use the last 'lags' returns from current_returns
        last_returns = current_returns.iloc[-lags:]
        features = pd.DataFrame({
            f'return_lag_{lag}': [last_returns.iloc[-lag]] for lag in range(1, lags + 1)
        }).T.squeeze()  # Transpose and squeeze to Series for predict
        
        # Reshape to 2D array as expected by XGBoost (1 sample, lags features)
        features_2d = np.array(features).reshape(1, -1)
        
        # 1-step forecast
        pred_var = model.predict(features_2d)[0]
        predictions.append(pred_var)
        
        # Current actual return
        actual_return = test_returns.loc[date]
        
        # Add the actual return to the current data (expanding window)
        current_returns = pd.concat([current_returns, pd.Series([actual_return], index=[date])])
        
        step += 1
        
        # Re-fit the model every 'reestimate_every' days on the expanded window
        if step % reestimate_every == 0:
            X_train, y_train = create_features_and_target_xgb(current_returns, lags=lags)
            model.fit(X_train, y_train)
    
    # Return predictions as pd.Series with test index
    return pd.Series(predictions, index=test_returns.index, name='predicted_variance')


def evaluate_hybrid_xgboost_with_garch(train_returns: pd.Series, test_returns: pd.Series, garch_params: dict,
                                       xgboost_params: dict, lags: int = 5, reestimate_every: int = 20) -> pd.Series:
    """
    Evaluation function for hybrid XGBoost model incorporating GARCH predictions as a feature.
    - Fits GARCH on the initial training set, obtains in-sample variances.
    - Prepares features including current GARCH variances and fits XGBoost.
    - For each day in the test set, first computes 1-step GARCH forecast (h_t), then uses it as a feature for XGBoost to predict squared returns on t.
    - Appends actual return and updates states.
    - Every 'reestimate_every' days, re-fits both models on the expanding window.

    Note: GARCH feature is now the current h_t (no additional lag), as it is computed from t-1 data and available for predicting r_t^2.
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

    objective = xgboost_params.get('objective', 'reg:squarederror')
    n_estimators = xgboost_params.get('n_estimators', 100)
    learning_rate = xgboost_params.get('learning_rate', 0.1)
    max_depth = xgboost_params.get('max_depth', 3)
    eval_metric = xgboost_params.get('eval_metric', 'rmse')

    # GARCH Forecaster initialization
    garch_forecaster = ARCH(p=p, o=o, q=q, dist=dist, vol=vol,
                            method=method, random_state=random_state, rescale=rescale)
    
    # XGBoost regressor initialization
    xgboost_model = XGBRegressor(objective=objective, n_estimators=n_estimators,
                                 learning_rate=learning_rate, max_depth=max_depth,
                                 eval_metric=eval_metric, random_state=random_state)
    
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
    
    # Initial data preparation for XGBoost using in-sample GARCH
    X_train, y_train = create_features_and_target_garch_xgb(current_returns, garch_in_sample, lags=lags)
    xgboost_model.fit(X_train, y_train)
    
    # List for predictions (XGBoost out-of-sample)
    predictions = []
    
    # Track out-of-sample GARCH predictions (start with in-sample)
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
        
        # Append GARCH prediction to current_garch_variances (now h_t is available)
        current_garch_variances = pd.concat([current_garch_variances, pd.Series([pred_h_garch], index=[date])])
        
        # For XGBoost prediction: prepare features using last lags from current_returns and current GARCH variance (h_t)
        last_returns = current_returns.iloc[-lags:]
        current_garch_var = pred_h_garch  # Use h_t directly
        
        features_dict = {f'return_lag_{lag}': last_returns.iloc[-lag] for lag in range(1, lags + 1)}
        features_dict['garch_variance'] = current_garch_var
        features = pd.Series(features_dict)
        
        # Reshape to 2D array
        features_2d = np.array(features).reshape(1, -1)
        
        # 1-step XGBoost forecast
        pred_var_xgb = xgboost_model.predict(features_2d)[0]
        predictions.append(pred_var_xgb)
        
        # Current actual return
        actual_return = test_returns.loc[date]
        
        # Update GARCH buffers with actual r_t for next step
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
            
            # Re-fit XGBoost with updated features including new in-sample GARCH (no shift)
            X_train, y_train = create_features_and_target_garch_xgb(current_returns, current_garch_variances, lags=lags)
            xgboost_model.fit(X_train, y_train)
    
    # Return XGBoost predictions as pd.Series with test index
    return pd.Series(predictions, index=test_returns.index, name='predicted_variance_hybrid')

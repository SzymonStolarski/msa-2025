import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arch import ARCH
from xgboost import XGBRegressor
import numpy as np

from functions.dataprep import create_features_and_target_garch_xgb, create_features_and_target_xgb

# def evaluate_garch(forecaster, train_returns: pd.Series, test_returns: pd.Series,
#                    reestimate_every: int = 20) -> pd.Series:
#     """
#     Evaluation function for the GARCH model using sktime.
#     - Fits the model on the initial training set.
#     - For each day in the test set, performs a 1-step variance forecast.
#     - Updates the filter (model state) based on the actual return without re-estimating parameters.
#     - Every 'reestimate_every' days (default 20), re-estimates parameters on the expanding window of data.

#     Assumes that train_returns and test_returns are pd.Series with date index (pd.DatetimeIndex).
#     Returns pd.Series with variance forecasts for the test_returns index.

#     Note: Uses ARCH from sktime, which relies on the 'arch' package underneath. The update method with update_params=False
#     allows updating the state (filter) without refitting parameters, simulating recursive conditional variance calculation.
#     """

#     # Copy of training data (expanding window)
#     current_data = train_returns.copy()
    
#     # Initial model fitting
#     forecaster.fit(current_data)
    
#     # List for predictions
#     predictions = []
    
#     # Step counter for re-estimation
#     step = 0
    
#     # Loop over test days
#     for date in test_returns.index:
#         print(date)
#         # 1-step variance forecast for the current day using ForecastingHorizon
#         fh = ForecastingHorizon(pd.PeriodIndex([date]), is_relative=False)
#         pred_var = forecaster.predict_var(fh=fh)
#         predictions.append(pred_var.iloc[0])  # Extract the variance value   
        
#         # Current actual return
#         actual_return = test_returns.loc[date]
#         # print(actual_return)
        
#         # Update the filter (state) without re-estimating parameters
#         forecaster.update(pd.Series([actual_return], index=[date]), update_params=False)
#         print(forecaster._y.tail(2))
        
#         # Add the actual return to the current data (expanding window)
#         current_data = pd.concat([current_data, pd.Series([actual_return], index=[date])])
        
#         step += 1
        
#         # Re-estimate parameters every 'reestimate_every' days on the expanded window
#         if step % reestimate_every == 0:
#             forecaster.fit(current_data)
    
#     return pd.Series(predictions, index=test_returns.index, name='predicted_variance')

GARCH_PARAMETERS = {
    'p': 1,
    'q': 1,
    'mean': 'Zero',
    'dist': "Normal",
    'vol': 'GARCH',
    'method': 'analytic',
    'random_state': 42,
    'rescale': False
}

GJRGARCH_PARAMETERS = {
    'p': 1,
    'o': 1,
    'q': 1,
    'mean': 'Zero',
    'dist': "Normal",
    'vol': 'GARCH',
    'method': 'analytic',
    'random_state': 42,
    'rescale': False
}


def evaluate_garch(train_returns: pd.Series, test_returns: pd.Series, garch_params: dict,
                   reestimate_every: int = 20) -> pd.Series:
    """
    Evaluation function for the GARCH model using sktime.
    - Fits the model on the initial training set.
    - For each day in the test set, performs a 1-step variance forecast.
    - Between re-estimations, manually recurses the GARCH variance using fixed parameters and previous state.
    - Every 'reestimate_every' days (default 20), re-estimates parameters on the expanding window of data and resets the state.

    Supports arbitrary p (ARCH lags) and q (GARCH lags).

    Assumes that train_returns and test_returns are pd.Series with date index (pd.DatetimeIndex or pd.PeriodIndex).
    Returns pd.Series with variance forecasts for the test_returns index.

    Note: This uses manual recursion to work around the update() method not fully updating the GARCH state with update_params=False.
    """
    p = garch_params.get('p', 1)
    q = garch_params.get('q', 1)
    mean = garch_params.get('mean')
    vol = garch_params.get('vol')
    dist = garch_params.get('dist')
    method = garch_params.get('method')
    random_state = garch_params.get('random_state')
    rescale = garch_params.get('rescale')

    # Forecaster initialization
    forecaster = ARCH(p=p, q=q, mean=mean, dist=dist, vol=vol,
                      method=method, random_state=random_state, rescale=rescale)
    # Copy of training data (expanding window)
    current_data = train_returns.copy()
    
    # Initial model fitting
    forecaster.fit(current_data)
    
    # Extract parameters and initial state from the underlying arch results
    results = forecaster._fitted_forecaster  # Access the fitted arch results
    params = results.params
    omega = params['omega']
    alphas = [params.get(f'alpha[{i}]', 0.0) for i in range(1, p + 1)]
    betas = [params.get(f'beta[{j}]', 0.0) for j in range(1, q + 1)]
    
    # Initial buffers: lists with oldest to newest
    last_r_sq_list = list(current_data.iloc[-p:]**2) if p > 0 else []
    last_h_list = list(results.conditional_volatility[-q:]**2) if q > 0 else []
    
    # Ensure buffers have exactly p and q elements (pad with 0 if data shorter than lags)
    if len(last_r_sq_list) < p:
        last_r_sq_list = [0.0] * (p - len(last_r_sq_list)) + last_r_sq_list
    if len(last_h_list) < q:
        last_h_list = [0.0] * (q - len(last_h_list)) + last_h_list
    
    # List for predictions
    predictions = []
    
    # Step counter for re-estimation
    step = 0
    
    # Loop over test days
    for date in test_returns.index:
        # 1-step variance forecast: manual recursion
        arch_part = sum(alphas[i-1] * last_r_sq_list[-i] for i in range(1, p + 1)) if p > 0 else 0.0
        garch_part = sum(betas[j-1] * last_h_list[-j] for j in range(1, q + 1)) if q > 0 else 0.0
        pred_h = omega + arch_part + garch_part
        predictions.append(pred_h)
        
        # Current actual return
        actual_return = test_returns.loc[date]
        
        # Update buffers: append new, remove oldest if full
        r_sq_new = actual_return**2
        if p > 0:
            last_r_sq_list.append(r_sq_new)
            if len(last_r_sq_list) > p:
                last_r_sq_list.pop(0)
        
        if q > 0:
            last_h_list.append(pred_h)
            if len(last_h_list) > q:
                last_h_list.pop(0)
        
        # Add the actual return to the current data (expanding window)
        current_data = pd.concat([current_data, pd.Series([actual_return], index=[date])])
        
        step += 1
        
        # Re-estimate parameters every 'reestimate_every' days on the expanded window
        if step % reestimate_every == 0:
            forecaster.fit(current_data)
            results = forecaster._fitted_forecaster
            params = results.params
            omega = params['omega']
            alphas = [params.get(f'alpha[{i}]', 0.0) for i in range(1, p + 1)]
            betas = [params.get(f'beta[{j}]', 0.0) for j in range(1, q + 1)]
            
            # Reset buffers from new fit
            last_r_sq_list = list(current_data.iloc[-p:]**2) if p > 0 else []
            last_h_list = list(results.conditional_volatility[-q:]**2) if q > 0 else []
            
            # Pad if necessary (though after fit on expanded data, should be full)
            if len(last_r_sq_list) < p:
                last_r_sq_list = [0.0] * (p - len(last_r_sq_list)) + last_r_sq_list
            if len(last_h_list) < q:
                last_h_list = [0.0] * (q - len(last_h_list)) + last_h_list
    
    # Return predictions as pd.Series with test index
    return pd.Series(predictions, index=test_returns.index, name='predicted_variance')

def evaluate_gjr_garch(train_returns: pd.Series, test_returns: pd.Series, gjrgarch_params: dict,
                   reestimate_every: int = 20) -> pd.Series:
    """
    Evaluation function for the GARCH model using sktime, supporting asymmetric variants like GJR-GARCH.
    - Fits the model on the initial training set.
    - For each day in the test set, performs a 1-step variance forecast.
    - Between re-estimations, manually recurses the GARCH variance using fixed parameters and previous state.
    - Every 'reestimate_every' days (default 20), re-estimates parameters on the expanding window of data and resets the state.

    Supports arbitrary p (ARCH lags), o (asymmetric lags for GJR), and q (GARCH lags).
    For standard GARCH, set o=0; for GJR-GARCH, set o>0 (typically o=1).

    Assumes that train_returns and test_returns are pd.Series with date index (pd.DatetimeIndex or pd.PeriodIndex).
    Returns pd.Series with variance forecasts for the test_returns index.

    Note: This uses manual recursion to work around the update() method not fully updating the GARCH state with update_params=False.
    """
    p = gjrgarch_params.get('p')
    q = gjrgarch_params.get('q')
    o = gjrgarch_params.get('o')
    mean = gjrgarch_params.get('mean')
    vol = gjrgarch_params.get('vol')
    dist = gjrgarch_params.get('dist')
    method = gjrgarch_params.get('method')
    random_state = gjrgarch_params.get('random_state')
    rescale = gjrgarch_params.get('rescale')

    # Forecaster initialization
    forecaster = ARCH(p=p, o=o, q=q, mean=mean, dist=dist, vol=vol,
                      method=method, random_state=random_state, rescale=rescale)
    
    # Copy of training data (expanding window)
    current_data = train_returns.copy()
    
    # Initial model fitting
    forecaster.fit(current_data)
    
    # Extract parameters and initial state from the underlying arch results
    results = forecaster._fitted_forecaster  # Access the fitted arch results
    params = results.params
    omega = params['omega']
    alphas = [params.get(f'alpha[{i}]', 0.0) for i in range(1, p + 1)]
    gammas = [params.get(f'gamma[{k}]', 0.0) for k in range(1, o + 1)]
    betas = [params.get(f'beta[{j}]', 0.0) for j in range(1, q + 1)]
    
    # Initial buffers: lists with oldest to newest
    last_r_sq_list = list(current_data.iloc[-p:]**2) if p > 0 else []
    last_neg_ind_r_sq_list = [(r**2 if r < 0 else 0.0) for r in current_data.iloc[-o:]] if o > 0 else []
    last_h_list = list(results.conditional_volatility[-q:]**2) if q > 0 else []
    
    # Ensure buffers have exactly p, o, q elements (pad with 0 if data shorter than lags)
    if len(last_r_sq_list) < p:
        last_r_sq_list = [0.0] * (p - len(last_r_sq_list)) + last_r_sq_list
    if len(last_neg_ind_r_sq_list) < o:
        last_neg_ind_r_sq_list = [0.0] * (o - len(last_neg_ind_r_sq_list)) + last_neg_ind_r_sq_list
    if len(last_h_list) < q:
        last_h_list = [0.0] * (q - len(last_h_list)) + last_h_list
    
    # List for predictions
    predictions = []
    
    # Step counter for re-estimation
    step = 0
    
    # Loop over test days
    for date in test_returns.index:
        # 1-step variance forecast: manual recursion
        arch_part = sum(alphas[i-1] * last_r_sq_list[-i] for i in range(1, p + 1)) if p > 0 else 0.0
        asym_part = sum(gammas[k-1] * last_neg_ind_r_sq_list[-k] for k in range(1, o + 1)) if o > 0 else 0.0
        garch_part = sum(betas[j-1] * last_h_list[-j] for j in range(1, q + 1)) if q > 0 else 0.0
        pred_h = omega + arch_part + asym_part + garch_part
        predictions.append(pred_h)
        
        # Current actual return
        actual_return = test_returns.loc[date]
        
        # Update buffers: compute new terms and append, remove oldest if full
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
            last_h_list.append(pred_h)
            if len(last_h_list) > q:
                last_h_list.pop(0)
        
        # Add the actual return to the current data (expanding window)
        current_data = pd.concat([current_data, pd.Series([actual_return], index=[date])])
        
        step += 1
        
        # Re-estimate parameters every 'reestimate_every' days on the expanded window
        if step % reestimate_every == 0:
            forecaster.fit(current_data)
            results = forecaster._fitted_forecaster
            params = results.params
            omega = params['omega']
            alphas = [params.get(f'alpha[{i}]', 0.0) for i in range(1, p + 1)]
            gammas = [params.get(f'gamma[{k}]', 0.0) for k in range(1, o + 1)]
            betas = [params.get(f'beta[{j}]', 0.0) for j in range(1, q + 1)]
            
            # Reset buffers from new fit
            last_r_sq_list = list(current_data.iloc[-p:]**2) if p > 0 else []
            last_neg_ind_r_sq_list = [(r**2 if r < 0 else 0.0) for r in current_data.iloc[-o:]] if o > 0 else []
            last_h_list = list(results.conditional_volatility[-q:]**2) if q > 0 else []
            
            # Pad if necessary (though after fit on expanded data, should be full)
            if len(last_r_sq_list) < p:
                last_r_sq_list = [0.0] * (p - len(last_r_sq_list)) + last_r_sq_list
            if len(last_neg_ind_r_sq_list) < o:
                last_neg_ind_r_sq_list = [0.0] * (o - len(last_neg_ind_r_sq_list)) + last_neg_ind_r_sq_list
            if len(last_h_list) < q:
                last_h_list = [0.0] * (q - len(last_h_list)) + last_h_list
    
    # Return predictions as pd.Series with test index
    return pd.Series(predictions, index=test_returns.index, name='predicted_variance')

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def calculate_metrics(df):
    """Calculate volatility (Parkinson) and liquidity (Amihud) metrics."""
    df = df.copy()
    df['sp_vol'] = np.sqrt(0.361 * (np.log(df['sp_high']) - np.log(df['sp_low']))**2)
    df['btc_vol'] = np.sqrt(0.361 * (np.log(df['btc_high']) - np.log(df['btc_low']))**2)

    df['sp_liq'] = np.abs(df['sp_close'].pct_change()) / (df['sp_volume'] * df['sp_close'])
    df['btc_liq'] = np.abs(df['btc_close'].pct_change()) / (df['btc_volume'] * df['btc_close'])

    return df.dropna()

def check_stationarity(series):
    """Check stationarity using Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return result[1] < 0.05  # True if stationary

def enforce_stationarity(df):
    """Ensure all series in a DataFrame are stationary, apply differencing if needed."""
    df_stationary = df.copy()
    for col in df.columns:
        if not check_stationarity(df[col]):
            df_stationary[col] = df[col].diff()
    return df_stationary.dropna()

def fit_var_model(df, maxlags=10):
    """Fit a VAR model and return the fitted model."""
    model = VAR(df)
    lag_order = model.select_order(maxlags=maxlags).aic
    results = model.fit(lag_order)
    return results

def generalized_variance_decomposition(model, steps=10):
    """Compute the generalized variance decomposition."""
    irf = model.irf(steps)
    irf_squared = (irf.irfs ** 2)
    sum_squared = irf_squared.sum(axis=0)
    gvd = sum_squared / sum_squared.sum(axis=1, keepdims=True)
    return gvd * 100  # Convert to percentages

def compute_spillovers(gvd_matrix):
    """Compute total and directional spillovers."""
    total_spillover = (gvd_matrix.sum() - np.diag(gvd_matrix).sum()) / gvd_matrix.sum() * 100

    sp_to_btc = gvd_matrix[1, 0] / gvd_matrix[1, :].sum() * 100
    btc_to_sp = gvd_matrix[0, 1] / gvd_matrix[0, :].sum() * 100

    return {
        "total_spillover": total_spillover,
        "sp_to_btc": sp_to_btc,
        "btc_to_sp": btc_to_sp
    }


# In[ ]:





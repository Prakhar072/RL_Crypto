#!/usr/bin/env python
# coding: utf-8

# # MAIN

# In[1]:


import pandas as pd
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
import import_ipynb
from Extraction_Module import FFCStack  # Import your FFCStack model
from FE import TemporalConvNet


# In[2]:


csv_files = glob.glob(r"C:\Users\Admin\Downloads\RL Datasets\testdata/*.csv")

df_dates = pd.DataFrame()
btc_arrays = pd.DataFrame()
sp500_arrays = pd.DataFrame()
arrays = []
max_n = 0

# Find the max row count (n)
for file_path in csv_files:
    df = pd.read_csv(file_path)
    max_n = max(max_n, df.shape[0])  # Get max number of rows

# Load and pad shorter arrays
for file_path in csv_files:
    df = pd.read_csv(file_path)
    df_dates = df.iloc[:,0]
    df = df.iloc[:, 1:]  # Remove the first column

    # Remove commas safely
    df = df.astype(str).apply(lambda col: col.str.replace(",", ""))

    # Convert "Vol." column (2nd last column)
    def convert_volume(val):
        val = str(val).replace(",", "")  # Remove commas
        if val.endswith("K"):
            return float(val[:-1]) * 1e3
        elif val.endswith("M"):
            return float(val[:-1]) * 1e6
        try:
            return float(val)  # Convert to float if it's a number
        except ValueError:
            return np.nan  # Set invalid values to NaN

    df.iloc[:, -2] = df.iloc[:, -2].map(convert_volume)

    # Convert "Change %" column (last column)
    df.iloc[:, -1] = df.iloc[:, -1].str.rstrip("%").astype(float)

    # Convert to NumPy array
    arr = df.values.astype(np.float32)

    # Ensure it has exactly 6 columns
    # if df.shape[1] != 6:
    #     print(f"Skipping {file_path}, incorrect columns: {df.shape[1]}")
    #     continue

    arr = df.values  # (n, 6)

    # Pad rows to (max_n, 6)
    pad_n = max_n - arr.shape[0]
    padded_arr = np.pad(arr, ((0, pad_n), (0, 0)), mode='constant')

    if "Bitcoin" in file_path:
        btc_arrays = padded_arr
    elif "SPY" in file_path:
        sp500_arrays = padded_arr

    arrays.append(padded_arr)

# Stack into (m, max_n, 6)
final_data = np.stack(arrays, axis=0)

print("Final stacked shape:", final_data.shape)  # (m, max_n, 6)


# In[3]:


sp500_df = pd.DataFrame(sp500_arrays)
btc_df = pd.DataFrame(btc_arrays)

btc_df_with_dates = pd.concat([df_dates.reset_index(drop=True), btc_df.reset_index(drop=True)], axis=1)
sp500_df_with_dates = pd.concat([df_dates.reset_index(drop=True), sp500_df.reset_index(drop=True)], axis=1)

# Optionally rename the date column
btc_df_with_dates.columns = ['Date'] + list(btc_df.columns)
sp500_df_with_dates.columns = ['Date'] + list(sp500_df.columns)

btc_df_with_dates = btc_df_with_dates.set_index('Date')
sp500_df_with_dates = sp500_df_with_dates.set_index('Date')

sp500_df_with_dates


# In[4]:


btc_df_with_dates


# In[5]:


btc_data = btc_df_with_dates.reset_index()
djia_data = sp500_df_with_dates.reset_index()
btc_data.columns = ['Date','btc_close','btc_open','btc_high','btc_low','btc_volume','btc_change%']
djia_data.columns = ['Date','sp_close','sp_open','sp_high','sp_low','sp_volume','sp_change%']

btc_clean = btc_data[(btc_data != 0).all(axis=1) &  btc_data.notnull().all(axis=1)] 
djia_clean = djia_data[(djia_data != 0).all(axis=1) & djia_data.notnull().all(axis=1)] 

min_len = min(len(djia_clean), len(btc_clean))
btc_trimmed = btc_clean.iloc[:min_len].reset_index(drop=True)
djia_trimmed = djia_clean.iloc[:min_len].reset_index(drop=True)
# Step 3: Concatenate side by side
voldata = pd.merge(djia_trimmed, btc_trimmed, on='Date')
voldata['Date'] = pd.to_datetime(voldata['Date'], format='%d-%m-%Y')
voldata.set_index('Date', inplace=True)
voldata = voldata.apply(pd.to_numeric, errors='coerce')
voldata = voldata
# voldata = pd.concat([djia_trimmed, btc_trimmed], axis=1).apply(pd.to_numeric, errors='coerce').iloc[:-1500]
voldata.info()


# In[6]:


voldata


# In[7]:


# print(len(voldata))
# print(voldata.index.min(), voldata.index.max())
# print(pd.date_range(start=voldata.index.min(), end=voldata.index.max(), freq='D').shape)


# In[8]:


# voldata.index = pd.to_datetime(voldata.index)

# # Create a complete daily date range from the min to the max date
# full_date_range = pd.date_range(start=voldata.index.min(), end=voldata.index.max(), freq='D')

# # Reindex voldata to this complete range
# voldata_padded = voldata.reindex(full_date_range, method='ffill')

# # Fill missing values (forward fill is usually sensible for financial time series)
# voldata_padded.index.freq = 'D'

# # Optional: confirm it's now properly padded
# print(voldata_padded.index.freq)  # Should now be 'D'


# In[9]:


# print(len(voldata_padded))


# In[10]:


# print(voldata_padded.index.to_series().diff().value_counts())


# In[11]:


# voldata_padded.iloc[500:]


# In[28]:


import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from spillover import (
    calculate_metrics, 
    enforce_stationarity, 
    fit_var_model, 
    generalized_variance_decomposition, 
    compute_spillovers
)
# Calculate volatility and liquidity
voldata = calculate_metrics(voldata)

# Example: choose volatility spillovers
metrics = voldata[['sp_vol', 'btc_vol']]

# Enforce stationarity
metrics_stationary = enforce_stationarity(metrics)
# print(metrics_stationary.info())
# metrics_stationary.index.freq = 'D'
# # correlation_matrix = metrics_stationary.corr()
# # print(correlation_matrix)
# print(metrics_stationary)
# print(metrics_stationary.index.to_series().diff().value_counts())

window_size = 150
results = pd.DataFrame(index=metrics_stationary.index[window_size-1:], columns=['Total Spillover', 'SP500 → BTC', 'BTC → SP500'])
for i in range(window_size, len(metrics_stationary)):

    window_data = metrics_stationary.iloc[i-window_size:i]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ValueWarning)
        # Fit VAR model
            var_model = fit_var_model(window_data)

            # Compute Generalized Variance Decomposition
            gvd_matrix = generalized_variance_decomposition(var_model)

            # Compute spillovers
            spillovers = compute_spillovers(gvd_matrix)

            # Save results
            results.iloc[i-window_size] = [
                spillovers['total_spillover'],
                spillovers['sp_to_btc'],
                spillovers['btc_to_sp']
            ]

    except Exception as e:
        print(f"Error at {metrics_stationary.index[i]}: {e}")
        results.iloc[i-window_size] = np.nan  # Fill with NaN on failure

print(results)
# Fit VAR model
# var_model = fit_var_model(metrics_stationary)

# # Compute Generalized Variance Decomposition
# gvd_matrix = generalized_variance_decomposition(var_model)

# # Compute spillovers
# spillovers = compute_spillovers(gvd_matrix)

# # Display results
# print(f"Total Spillover Index: {spillovers['total_spillover']:.2f}%")
# print(f"SP500 → BTC: {spillovers['sp_to_btc']:.2f}%")
# print(f"BTC → SP500: {spillovers['btc_to_sp']:.2f}%")


# In[27]:


results.info()


# In[33]:


spillovers_smoothed = results.rolling(window=30).mean()


# In[34]:


# Optionally, plot the spillover index over time
import matplotlib.pyplot as plt

spillovers_smoothed['Total Spillover'].plot(title='Rolling Spillover Index (150-day window)', figsize=(12, 6))
plt.ylabel('Spillover (%)')
plt.xlabel('Date')
plt.grid(True)
plt.show()


# In[15]:


# Convert all valid numbers, set invalid values to NaN
final_data = np.array(final_data, dtype=np.float32)  # Ensure entire array is float32

# Replace NaNs (if any) with 0 before conversion
final_data = np.nan_to_num(final_data, nan=0.0)

# Convert to tensor
final_data_tensor = torch.tensor(final_data, dtype=torch.float32)
final_data_tensor = final_data_tensor.permute(0, 2, 1)

print("Tensor shape:", final_data_tensor.shape)  # Should match your expectations

batch_size = 8  # Adjust as needed
seq_len = final_data.shape[1]  # max_n (time steps)
input_dim = final_data.shape[2]  # 7 (features)
num_channels = 16  # Use a list
heads=4
ffc_model = FFCStack(input_dim,num_channels,heads)

# Pass a batch through TCN to check output shape
# sample_input = final_data_tensor[:batch_size].permute(0, 2, 1)

# Now pass it to the TCN
output = ffc_model(final_data_tensor)

print("FFC Output shape:", output.shape)  # Check if this is as expected
n_assets = output.shape[0];


# In[16]:


from RL import *
# Now run the pipeline with tensor data
# Assuming your FFC output is a tensor in variable 'output' with shape (3, 1, 3733)
# Convert tensor to correct format for RL processing

# First, squeeze to remove the singleton dimension (3, 1, 3733) -> (3, 3733)
processed_data = output.squeeze(1).permute(1, 0).detach().clone()  # (3733, 3)
processed_data = (processed_data - processed_data.mean(dim=0)) / (processed_data.std(dim=0) + 1e-6)
env = PortfolioEnv(processed_data)  # Convert to numpy array

# Train agent
trained_agent = train_agent(env,n_assets,episodes=100)

# Evaluation and visualization remain the same
asset_weights = evaluate_agent(trained_agent, env)
# Visualization


# In[ ]:


copy_weights = asset_weights

step = 100  # You can use 3732//10 too
x = np.arange(0, 3732, step)
asset_0 = copy_weights[::step, 0]
asset_1 = copy_weights[::step, 1]
asset_2 = copy_weights[::step, 2]
asset_3 = copy_weights[::step, 3]
asset_4 = copy_weights[::step, 4]

plt.figure(figsize=(12, 6))
plt.stackplot(x, asset_0, asset_1, asset_2, asset_3, asset_4, labels=["BTC", "DJIA", "ETH", "GLDF", "USDT"], alpha=0.8)
plt.title("Downsampled Stacked Area Chart")
plt.xlabel("Time Step")
plt.ylabel("Allocation")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


import os
import torch
import numpy as np

def ensure_directory_exists(filepath):
    """Ensure the directory for the file exists"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_model(agent, config, weights_history, filepath):
    """Safely save the model with proper serialization"""
    ensure_directory_exists(filepath)

    save_data = {
        'model_state': agent.policy.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'config': {
            'n_assets': config['n_assets'],
            'gamma': config['gamma'],
            'input_normalization': {
                'mean': config['input_normalization']['mean'].cpu().numpy().tolist(),
                'std': config['input_normalization']['std'].cpu().numpy().tolist()
            }
        },
        'weights_history': weights_history.tolist() if isinstance(weights_history, np.ndarray) else weights_history
    }

    torch.save(save_data, filepath, _use_new_zipfile_serialization=True)
    print(f"Model successfully saved to {filepath}")

def load_model(filepath, device='cpu'):
    """Safely load the model with proper deserialization"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")

    try:
        data = torch.load(filepath, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Warning: Safe loading failed ({str(e)}). Attempting unsafe load (only use with trusted sources)")
        data = torch.load(filepath, map_location=device, weights_only=False)

    agent = RLAgent(n_assets=data['config']['n_assets'])
    agent.policy.load_state_dict(data['model_state'])
    agent.optimizer.load_state_dict(data['optimizer_state'])
    agent.gamma = data['config']['gamma']

    norm_params = {
        'mean': torch.tensor(data['config']['input_normalization']['mean'], device=device),
        'std': torch.tensor(data['config']['input_normalization']['std'], device=device)
    }

    weights_history = np.array(data['weights_history'])

    print(f"Model successfully loaded from {filepath}")
    return agent, norm_params, weights_history

# Example usage:
if __name__ == "__main__":
    # After training your model...
    save_config = {
        'n_assets': 5,
        'gamma': trained_agent.gamma,
        'input_normalization': {
            'mean': processed_data.mean(dim=0),
            'std': processed_data.std(dim=0)
        }
    }

    # Save the model
    model_path = "saved_models/portfolio_rl_safe.pth"
    save_model(trained_agent, save_config, asset_weights, model_path)

    # Later, when you want to load it...
    try:
        loaded_agent, loaded_norm, loaded_weights = load_model(model_path)

        # Test the loaded model
        test_state = env.reset()
        action, _ = loaded_agent.get_action(test_state)
        print("Test allocation from loaded model:", action)
    except Exception as e:
        print(f"Error loading model: {str(e)}")


# In[ ]:


original_action, _ = trained_agent.get_action(test_state)
loaded_action, _ = loaded_agent.get_action(test_state)

print("Original model weights:", original_action)
print("Loaded model weights:", loaded_action)
print("Difference:", np.abs(original_action - loaded_action))


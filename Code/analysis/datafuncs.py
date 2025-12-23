## Functions for use in data processing noteboooks

# import dependencies
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict

#------------------------------------------------------------------------
# Processing an individual mouse's eyeblink data
#------------------------------------------------------------------------

# Group stim files by date
def group_by_date(fec_files, stim_files):
    """Group files into dict with experiment date as the key and a list of filenames matching that date as the value

    Returns:
        files_by_date (dict): key: date; val: list of filenames from that date
    """
    
    files_by_date = defaultdict(list)
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')

    for file in fec_files + stim_files:
        match = date_pattern.search(file)
        if match:
            date_str = match.group(1)
            files_by_date[date_str].append(file)
    
    return files_by_date


# Func to process each day's files
def process_date_files(files_by_date, date):
    """Process files for a specific date, returning FEC and stimulus dataframes.

    Args:
        date (str): Date string

    Returns:
        data (tuple): 2 dataframes containing FEC and stimulus data, respectively
    """
    
    fec_files = [file for file in files_by_date[date] if 'fec' in file]
    stim_files = [file for file in files_by_date[date] if 'stim' in file]
    
    print(f"Processing {date} with {len(fec_files)} FEC files and {len(stim_files)} stimulus files.")
    
    # Check if there are multiple files for fec and stim on this date
    if len(fec_files) == 1 & len(stim_files) == 1:
        df_fec, df_stim = pd.read_csv(fec_files[0]), pd.read_csv(stim_files[0])
        return df_fec, df_stim
    return None


# Normalize FEC values to [0, 100]
def normalize_fec(fec_series):
    """Normalize FEC values to a range of [0, 100].

    Args:
        fec_series (pandas.Series): Pandas series containing raw FEC data

    Returns:
        pd.Series: Pandas series containing FEC data normalized to fall between [0, 100]
    """
    fec_min = fec_series.min()
    fec_max = fec_series.max()
    if 1. < fec_max <= 100.:
        return fec_series
    return round(((fec_series - fec_min) / (fec_max - fec_min)) * 100)


# Combine all data into one dataframe
def merge_dataframes(df_fec, df_stim):
    """Merge fec and stim data onto a single dataframe

    Args:
        df_fec (pd.DataFrame): Dataframe containing FEC data
        df_stim (pd.DataFrame): Dataframe containing CS timestamps

    Returns:
        pd.DataFrame: Dataframe with fec and stimulus data, as well as time elapsed since the current trial's CS ("Relative Timestamp", measured in ms)
    """
    # Merge the dataframes based on the 'Trial #' column
    merged_df = pd.merge(df_fec, df_stim, on='Trial #', how='left')

    # # Calculate timestamps relative to CS onset
    merged_df["Timestamp"] = pd.to_datetime(merged_df["Timestamp"]) # Convert Timestamp from string so it can be subtracted
    merged_df["CS Timestamp"] = pd.to_datetime(merged_df["CS Timestamp"]) # Convert Timestamp from string so it can be subtracted
    merged_df["Relative Timestamp"] = pd.to_timedelta(merged_df["Timestamp"] - merged_df["CS Timestamp"])  # calc and convert to timedeltas
    merged_df["Relative Timestamp"] = merged_df["Relative Timestamp"].dt.total_seconds() * 1000  # convert to milliseconds
    
    return merged_df


# Calculate mean curve and interpolated approximations of individual trial curves
def interpolated_curve(files_by_date, date, x_common):
    """Generate interpolated mean curve and individua curves for all trials on a given date

    Args:
        files_by_date (dict): files_by_date (dict): key: date; val: list of filenames from that date
        date (str): date whose mean to calculate
        x_common (np.ndarray): x values to interpolate curves onto in order to take the mean

    Returns:
        tuple: 1-D array of y-values of mean curve; 2-D array (num_trials x num_interpolated_pts) of interpolated y-values of individual trials
    """
    # Retrieve fec and stim data
    dfs = process_date_files(files_by_date, date)
    if dfs is None:
        return
    else:
        df_fec, df_stim = dfs
    
    num_trials = len(df_stim)  # save number of trials
    df_fec["FEC"] = normalize_fec(df_fec["FEC"])  # Normalize FEC values
    
    # Merge the dataframes based on the 'Trial #' column
    merged_df = merge_dataframes(df_fec, df_stim)
        
    # Interpolate each y onto x_common
    y_interps = np.zeros((num_trials, len(x_common)))  # Initialize array to hold interpolated values
    for trial in range(1, num_trials + 1):  # Loop through trials 1 to 9
        filter = (merged_df['Trial #'] == trial) & (merged_df['Airpuff'] == True) & (-50 <= merged_df['Relative Timestamp']) & (merged_df['Relative Timestamp'] <= 1500)  # Filter for trial i only (if it has an airpuff)
        df_i = merged_df[filter]
        if not df_i.empty:
            y_interp = np.interp(x_common, df_i['Relative Timestamp'], df_i['FEC'])
            y_interps[trial - 1, :] = y_interp  # Store interpolated values

    # Calculate mean across trials
    y_mean = np.mean(y_interps, axis=0)
    return y_mean, y_interps
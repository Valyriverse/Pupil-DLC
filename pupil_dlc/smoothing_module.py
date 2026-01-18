#!/usr/bin/env python
"""
Smoothing module for Pupil-DLC Pipeline
Provides various smoothing algorithms for post-processing ellipse fitting results
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import warnings

def filter_by_rate_of_change(df, lower_perc=5, upper_perc=95):
    """
    Filters the session data by calculating the rate of change and then applying
    percentile-based filtering to remove outliers.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing pupil tracking data
    lower_perc : float
        The lower percentile threshold for filtering (default: 5)
    upper_perc : float
        The upper percentile threshold for filtering (default: 95)

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with filtered values based on the rate of change within the specified percentiles.
    """
    df_filtered = df.copy()
    
    # Check if required columns exist
    if 'Time_Frames' not in df_filtered.columns or 'Largest_Radius' not in df_filtered.columns:
        warnings.warn("Required columns 'Time_Frames' or 'Largest_Radius' not found. Returning original DataFrame.")
        return df_filtered
    
    # Remove any NaN values for calculation
    valid_mask = ~(df_filtered['Time_Frames'].isna() | df_filtered['Largest_Radius'].isna())
    if not valid_mask.any():
        warnings.warn("No valid data points found. Returning original DataFrame.")
        return df_filtered
    
    # Calculate the differences between successive time points and radius values
    time_diffs = np.diff(df_filtered['Time_Frames'])
    radius_diffs = np.diff(df_filtered['Largest_Radius'])
    
    # Avoid division by zero
    time_diffs[time_diffs == 0] = 1e-10
    
    # Compute the instantaneous rate of change (approximate derivative)
    rate_of_change = np.insert(radius_diffs / time_diffs, 0, 0)  # Insert 0 at the beginning
    
    df_filtered['Rate_of_Change'] = rate_of_change
    
    # Calculate the lower and upper bounds for rate of change
    valid_rates = df_filtered['Rate_of_Change'].dropna()
    if len(valid_rates) == 0:
        warnings.warn("No valid rate of change values. Returning original DataFrame.")
        return df_filtered.drop('Rate_of_Change', axis=1)
    
    lower_bound = np.percentile(valid_rates, lower_perc)
    upper_bound = np.percentile(valid_rates, upper_perc)
    
    # Filter the DataFrame based on the calculated bounds
    filter_mask = (df_filtered['Rate_of_Change'] <= upper_bound) & (df_filtered['Rate_of_Change'] >= lower_bound)
    df_filtered = df_filtered[filter_mask].copy()
    
    # Remove the temporary Rate_of_Change column
    df_filtered = df_filtered.drop('Rate_of_Change', axis=1)
    
    # Reset index to maintain continuity
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered

def moving_average_smooth(data, window_size=5):
    """
    Apply moving average smoothing to data.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    window_size : int
        Size of the moving average window (default: 5)
        
    Returns:
    --------
    numpy.ndarray
        Smoothed data
    """
    if len(data) < window_size:
        warnings.warn(f"Data length ({len(data)}) is smaller than window size ({window_size}). Using data length as window size.")
        window_size = len(data)
    
    # Convert to numpy array and handle NaN values
    data_array = np.array(data, dtype=float)
    
    # Simple moving average
    smoothed = np.convolve(data_array, np.ones(window_size)/window_size, mode='same')
    
    # Handle edge effects
    half_window = window_size // 2
    for i in range(half_window):
        # Beginning
        smoothed[i] = np.nanmean(data_array[:i+half_window+1])
        # End
        smoothed[-(i+1)] = np.nanmean(data_array[-(i+half_window+1):])
    
    return smoothed

def gaussian_smooth(data, sigma=1.0):
    """
    Apply Gaussian smoothing to data.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    sigma : float
        Standard deviation for Gaussian kernel (default: 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Smoothed data
    """
    data_array = np.array(data, dtype=float)
    
    # Handle NaN values by interpolating first
    valid_mask = ~np.isnan(data_array)
    if not np.any(valid_mask):
        return data_array
    
    # Linear interpolation for NaN values
    if np.any(~valid_mask):
        indices = np.arange(len(data_array))
        data_array[~valid_mask] = np.interp(
            indices[~valid_mask], 
            indices[valid_mask], 
            data_array[valid_mask]
        )
    
    # Apply Gaussian filter
    smoothed = gaussian_filter1d(data_array, sigma=sigma)
    
    return smoothed

def savitzky_golay_smooth(data, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay smoothing to data.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    window_length : int
        Length of the filter window (must be odd, default: 11)
    polyorder : int
        Order of the polynomial used for fitting (default: 3)
        
    Returns:
    --------
    numpy.ndarray
        Smoothed data
    """
    data_array = np.array(data, dtype=float)
    
    # Ensure window_length is odd and valid
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(data_array):
        window_length = len(data_array) if len(data_array) % 2 == 1 else len(data_array) - 1
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    # Handle NaN values
    valid_mask = ~np.isnan(data_array)
    if not np.any(valid_mask):
        return data_array
    
    if np.any(~valid_mask):
        indices = np.arange(len(data_array))
        data_array[~valid_mask] = np.interp(
            indices[~valid_mask], 
            indices[valid_mask], 
            data_array[valid_mask]
        )
    
    # Apply Savitzky-Golay filter
    try:
        smoothed = signal.savgol_filter(data_array, window_length, polyorder)
    except ValueError as e:
        warnings.warn(f"Savitzky-Golay filter failed: {e}. Falling back to moving average.")
        smoothed = moving_average_smooth(data_array, window_size=5)
    
    return smoothed

def butterworth_smooth(data, cutoff_freq=0.1, sampling_rate=1.0, order=4):
    """
    Apply Butterworth low-pass filter smoothing to data.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    cutoff_freq : float
        Cutoff frequency as fraction of Nyquist frequency (default: 0.1)
    sampling_rate : float
        Sampling rate of the data (default: 1.0)
    order : int
        Filter order (default: 4)
        
    Returns:
    --------
    numpy.ndarray
        Smoothed data
    """
    data_array = np.array(data, dtype=float)
    
    # Handle NaN values
    valid_mask = ~np.isnan(data_array)
    if not np.any(valid_mask):
        return data_array
    
    if np.any(~valid_mask):
        indices = np.arange(len(data_array))
        data_array[~valid_mask] = np.interp(
            indices[~valid_mask], 
            indices[valid_mask], 
            data_array[valid_mask]
        )
    
    # Design Butterworth filter
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    
    if normal_cutoff >= 1.0:
        warnings.warn("Cutoff frequency too high, reducing to 0.45 * Nyquist")
        normal_cutoff = 0.45
    
    try:
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        smoothed = signal.filtfilt(b, a, data_array)
    except ValueError as e:
        warnings.warn(f"Butterworth filter failed: {e}. Falling back to Gaussian smoothing.")
        smoothed = gaussian_smooth(data_array, sigma=1.0)
    
    return smoothed

def adaptive_smooth(data, method='auto', **kwargs):
    """
    Apply adaptive smoothing based on data characteristics.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    method : str
        Smoothing method ('auto', 'moving_average', 'gaussian', 'savgol', 'butterworth')
    **kwargs : dict
        Additional parameters for specific smoothing methods
        
    Returns:
    --------
    numpy.ndarray
        Smoothed data
    """
    data_array = np.array(data, dtype=float)
    
    if method == 'auto':
        # Choose method based on data characteristics
        data_length = len(data_array)
        noise_level = np.nanstd(np.diff(data_array)) if data_length > 1 else 0
        
        if data_length < 50:
            method = 'moving_average'
        elif noise_level > np.nanstd(data_array) * 0.5:
            method = 'savgol'  # Better for noisy data
        else:
            method = 'gaussian'  # Good general purpose
    
    # Apply selected method
    if method == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        return moving_average_smooth(data_array, window_size)
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_smooth(data_array, sigma)
    elif method == 'savgol':
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 3)
        return savitzky_golay_smooth(data_array, window_length, polyorder)
    elif method == 'butterworth':
        cutoff_freq = kwargs.get('cutoff_freq', 0.1)
        sampling_rate = kwargs.get('sampling_rate', 1.0)
        order = kwargs.get('order', 4)
        return butterworth_smooth(data_array, cutoff_freq, sampling_rate, order)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

def smooth_pupil_data(df, columns_to_smooth=None, method='auto', save_original=True, **kwargs):
    """
    Smooth pupil tracking data in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing pupil tracking data with columns:
        (Time_Frames, X_Position, Y_Position, Radius_1, Radius_2, Angle, Largest_Radius, Eye_Diameter)
    columns_to_smooth : list or None
        List of column names to smooth. If None, defaults to ['Largest_Radius']
    method : str
        Smoothing method to use
    save_original : bool
        Whether to keep original columns (with '_original' suffix)
    **kwargs : dict
        Additional parameters for smoothing methods
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with smoothed data
    """
    df_smooth = df.copy()
    
    # Determine columns to smooth
    if columns_to_smooth is None:
        # Default to smoothing only the Largest_Radius (pupil size)
        columns_to_smooth = ['Largest_Radius']
        
        # Verify the column exists
        if 'Largest_Radius' not in df_smooth.columns:
            # Fallback to available pupil measurement columns
            available_cols = df_smooth.columns.tolist()
            pupil_measurement_cols = [col for col in available_cols 
                                    if col in ['Largest_Radius', 'Eye_Diameter', 'Radius_1', 'Radius_2']]
            if pupil_measurement_cols:
                columns_to_smooth = [pupil_measurement_cols[0]]  # Use the first available
                warnings.warn(f"Largest_Radius not found. Using {pupil_measurement_cols[0]} instead.")
            else:
                raise ValueError("No suitable pupil measurement columns found for smoothing.")
    
    # Apply smoothing
    for col in columns_to_smooth:
        if col in df_smooth.columns:
            # Save original if requested
            if save_original:
                df_smooth[f"{col}_original"] = df_smooth[col].copy()
            
            # Apply smoothing
            smoothed_data = adaptive_smooth(df_smooth[col].values, method=method, **kwargs)
            df_smooth[f"{col}_smoothed"] = smoothed_data
            
            # Replace original column with smoothed data
            df_smooth[col] = smoothed_data
        else:
            warnings.warn(f"Column '{col}' not found in DataFrame. Skipping.")
    
    return df_smooth
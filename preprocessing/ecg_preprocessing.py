import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(signal, cutoff=40, fs=100, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    
    if std == 0:
        return signal
    
    return (signal - mean) / std


def preprocess_ecg(signal):
    """
    Full ECG preprocessing pipeline
    """
    
    # Step 1: Denoising
    signal = butter_lowpass_filter(signal)
    
    # Step 2: Normalization
    signal = normalize_signal(signal)
    
    return signal
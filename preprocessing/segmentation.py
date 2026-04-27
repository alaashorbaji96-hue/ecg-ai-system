import numpy as np


def segment_signal(signal, window_size=200, step_size=200):
    """
    Segment a single ECG signal into fixed-size windows
    
    Args:
        signal (array): 1D ECG signal
        window_size (int): size of each segment
        step_size (int): step between segments
    
    Returns:
        numpy array: segmented signal windows
    """
    
    segments = []
    
    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        segment = signal[start:end]
        segments.append(segment)
    
    return np.array(segments)
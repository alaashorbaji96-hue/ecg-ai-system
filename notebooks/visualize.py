import matplotlib.pyplot as plt
from utils.helpers import load_dataset
from preprocessing.ecg_preprocessing import preprocess_ecg
from preprocessing.segmentation import segment_signal


def visualize():
    
    # Load small sample
    signals = load_dataset("data/ptb-xl", 100, limit=1)
    
    signal = signals[0]
    
    # أول lead
    raw = signal[:, 0]
    
    # Preprocess
    processed = preprocess_ecg(raw)
    
    # Segments
    segments = segment_signal(processed, 200, 200)
    
    # =========================
    # Plot
    # =========================
    
    plt.figure(figsize=(15, 10))
    
    # Raw signal
    plt.subplot(3, 1, 1)
    plt.plot(raw)
    plt.title("Raw ECG Signal")
    
    # Processed
    plt.subplot(3, 1, 2)
    plt.plot(processed)
    plt.title("Processed ECG Signal")
    
    # Segment example
    plt.subplot(3, 1, 3)
    plt.plot(segments[0])
    plt.title("Segmented ECG (1 window)")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
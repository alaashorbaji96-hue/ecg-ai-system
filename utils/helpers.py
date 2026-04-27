import os
import wfdb


def load_ecg_signal(file_path):
    """
    Load ECG signal using WFDB
    """
    record = wfdb.rdsamp(file_path)
    signal = record[0]
    return signal


def load_dataset(data_path, sampling_rate=100, limit=None, verbose=True):
    """
    Load PTB-XL dataset signals + correct IDs
    
    Returns:
        signals (list)
        ids (list)
    """
    
    folder = "records100" if sampling_rate == 100 else "records500"
    full_path = os.path.join(data_path, folder)
    
    signals = []
    ids = []
    count = 0
    
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file.endswith(".dat"):
                
                file_id = file[:-4]
                file_path = os.path.join(root, file_id)
                
                try:
                    signal = load_ecg_signal(file_path)
                    
                    # 🔥 الحل المهم: ID مطابق للـ metadata
                    relative_path = os.path.relpath(file_path, data_path)
                    relative_path = relative_path.replace("\\", "/")
                    
                    signals.append(signal)
                    ids.append(relative_path)
                    
                    count += 1
                    
                    if verbose:
                        print(f"Loaded {count}")
                    
                    if limit is not None and count >= limit:
                        return signals, ids
                
                except Exception as e:
                    if verbose:
                        print(f"Error loading {file_path}: {e}")
                    continue
    
    return signals, ids


# ===============================
# TEST RUN
# ===============================
if __name__ == "__main__":
    
    print("Loading ECG dataset...")
    
    signals, ids = load_dataset(
        data_path="data/ptb-xl",
        sampling_rate=100,
        limit=5
    )
    
    print(f"\n✅ Loaded signals: {len(signals)}")
    print(f"📌 Example ID: {ids[0]}")
    
    if len(signals) > 0:
        print(f"📊 Shape: {signals[0].shape}")
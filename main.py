from utils.helpers import load_dataset
from preprocessing.ecg_preprocessing import preprocess_ecg
from preprocessing.segmentation import segment_signal

from preprocessing.metadata_preprocessing import (
    load_metadata,
    parse_scp_codes,
    extract_labels,
    encode_labels
)

import numpy as np


def build_dataset():
    
    print("🚀 Loading signals...")
    signals, signal_ids = load_dataset("data/ptb-xl", 100, limit=1000)
    
    print("📊 Loading metadata...")
    df = load_metadata("data/ptb-xl")
    df = parse_scp_codes(df)
    
    # 🔥 نحط index حسب اسم الملف
    df = df.set_index("filename_lr")
    
    # استخراج labels
    labels = extract_labels(df, min_samples=1000)
    y_all = encode_labels(df, labels)
    
    print("🧠 Processing signals + labels...")
    
    X = []
    y = []
    
    for i, signal in enumerate(signals):
        
        signal_id = signal_ids[i]
        
        # تأكد الإشارة موجودة بالـ metadata
        if signal_id not in df.index:
            print(f"⚠️ Skipping {signal_id} (not found in metadata)")
            continue
        
        # جلب label الصحيح
        label_index = df.index.get_loc(signal_id)
        label = y_all[label_index]
        
        # أخذ lead واحد
        lead = signal[:, 0]
        
        # preprocessing
        processed = preprocess_ecg(lead)
        
        # segmentation
        segments = segment_signal(processed, 200, 200)
        
        for seg in segments:
            X.append(seg)
            y.append(label)
        
        print(f"✔ Signal {i+1} → {len(segments)} segments")
    
    # تحويل لنماذج
    X = np.array(X)
    X = X[..., np.newaxis]  # مهم للـ CNN
    y = np.array(y)
    
    print("\n✅ Final Dataset Ready")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y


if __name__ == "__main__":
    X, y = build_dataset()
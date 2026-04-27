import pandas as pd
import numpy as np
import ast


# =========================
# Load Metadata
# =========================
def load_metadata(data_path):
    df = pd.read_csv(f"{data_path}/ptbxl_database.csv")
    return df


# =========================
# Parse scp_codes
# =========================
def parse_scp_codes(df):
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    return df


# =========================
# 🔥 Extract Labels (محسّن)
# =========================
def extract_labels(df, min_samples=200):
    """
    Select most frequent labels only
    """
    
    label_counts = {}
    
    # Count frequency
    for codes in df['scp_codes']:
        for key in codes.keys():
            label_counts[key] = label_counts.get(key, 0) + 1
    
    # Sort labels by frequency (important 🔥)
    label_counts = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Filter
    selected_labels = [k for k, v in label_counts.items() if v >= min_samples]
    
    print("\n📊 Label Distribution (Top 10):")
    for k, v in list(label_counts.items())[:10]:
        print(f"{k}: {v}")
    
    print(f"\n✅ Selected {len(selected_labels)} labels:")
    print(selected_labels)
    
    return selected_labels


# =========================
# 🔥 Encode Labels (محسّن)
# =========================
def encode_labels(df, selected_labels):
    """
    Convert labels to multi-hot encoding (NumPy)
    """
    
    y = []
    
    for codes in df['scp_codes']:
        row = np.zeros(len(selected_labels))
        
        for i, label in enumerate(selected_labels):
            if label in codes:
                row[i] = 1
        
        y.append(row)
    
    y = np.array(y)
    
    print(f"\n📊 Label Matrix Shape: {y.shape}")
    
    return y


# =========================
# 🔥 Helper: Label Stats
# =========================
def print_label_stats(y, labels):
    print("\n📊 Label Stats:")
    
    counts = y.sum(axis=0)
    
    for i, label in enumerate(labels):
        print(f"{label}: {int(counts[i])}")


# =========================
# TEST
# =========================
if __name__ == "__main__":
    
    df = load_metadata("data/ptb-xl")
    df = parse_scp_codes(df)
    
    labels = extract_labels(df, min_samples=1000)
    y = encode_labels(df, labels)
    
    print_label_stats(y, labels)
    
    print("\n🔍 Example:")
    print("Labels:", labels[:5])
    print("Encoded:", y[0][:5])
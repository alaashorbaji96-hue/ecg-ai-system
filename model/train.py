import numpy as np
from model.model import build_model
from main import build_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


def train():
    
    # =========================
    # Load Data
    # =========================
    X, y = build_dataset()
    
    # =========================
    # Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # =========================
    # 🔥 Compute Class Weights
    # =========================
    pos_weights = ((y_train.shape[0] - y_train.sum(axis=0)) / (y_train.sum(axis=0) + 1e-6)) ** 0.5
    
    print("Building model...")
    
    model = build_model(
        input_shape=(X.shape[1], 1),
        num_classes=y.shape[1],
        pos_weights=pos_weights
    )
    
    model.summary()
    
    print("Training...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=32
    )
    
    # =========================
    # Save model
    # =========================
    model.save("model/ecg_model.keras")
    print("✅ Model saved!")
    
    return model, X_test, y_test


# =========================
# 🔥 Threshold Evaluation (مهم جدًا)
# =========================
def evaluate_thresholds(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    thresholds = [0.5, 0.3, 0.2]
    
    for t in thresholds:
        y_bin = (y_pred > t).astype(int)
        
        precision = precision_score(y_test, y_bin, average='micro', zero_division=0)
        recall = recall_score(y_test, y_bin, average='micro', zero_division=0)
        f1 = f1_score(y_test, y_bin, average='micro', zero_division=0)
        
        print("\n==========================")
        print(f"Threshold = {t}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    
    model, X_test, y_test = train()
    
    print("\n🔥 Evaluating thresholds...")
    evaluate_thresholds(model, X_test, y_test)
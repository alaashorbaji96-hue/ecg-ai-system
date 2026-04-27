import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Dense, Dropout, Input, Bidirectional,
    Attention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model


# =========================
# Weighted Loss
# =========================
def weighted_loss(pos_weights):
    
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * pos_weights + (1 - y_true)
        return tf.reduce_mean(bce * weight_vector)
    
    return loss


# =========================
# 🔥 Model with Attention
# =========================
def build_model(input_shape, num_classes, pos_weights):
    
    inputs = Input(shape=input_shape)
    
    # CNN
    x = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # 🔥 BiLSTM (مهم return_sequences=True)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    # =========================
    # 🔥 Attention Layer
    # =========================
    attention = Attention()([x, x])
    
    # =========================
    # Pooling
    # =========================
    x = GlobalAveragePooling1D()(attention)
    
    # Dense
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_loss(pos_weights),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_acc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return model
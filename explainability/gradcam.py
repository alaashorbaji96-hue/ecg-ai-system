import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================
# 🔥 Grad-CAM (1D ECG)
# =========================
def compute_gradcam(model, signal, class_index):
    """
    Compute Grad-CAM for 1D ECG signal
    """
    
    # تأكد الشكل (1, 200, 1)
    signal = np.expand_dims(signal, axis=0)
    
    # 🔥 نجيب آخر Conv1D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv1d" in layer.name.lower():
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("No Conv1D layer found in model.")
    
    # model يعطي conv output + prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(signal)
        loss = predictions[:, class_index]
    
    grads = tape.gradient(loss, conv_outputs)
    
    # 🔥 average gradients
    pooled_grads = tf.reduce_mean(grads, axis=1)
    
    conv_outputs = conv_outputs[0]     # (time, channels)
    pooled_grads = pooled_grads[0]    # (channels,)
    
    # 🔥 weighted combination
    cam = np.zeros(conv_outputs.shape[0])
    
    for i in range(pooled_grads.shape[-1]):
        cam += pooled_grads[i] * conv_outputs[:, i]
    
    # ReLU + normalization
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    
    return cam


# =========================
# 🔥 Visualization (SCATTER — الأفضل)
# =========================
def plot_gradcam(signal, cam):
    """
    Plot ECG with clear Grad-CAM scatter heatmap
    """
    
    signal = signal.squeeze()
    
    # 🔥 Resize CAM → match signal length
    cam_resized = np.interp(
        np.linspace(0, len(cam), num=len(signal)),
        np.arange(len(cam)),
        cam
    )
    
    # 🔥 Boost contrast (أهم سطر)
    cam_resized = cam_resized ** 2
    
    plt.figure(figsize=(14, 4))
    
    # ECG line
    plt.plot(signal, color='black', linewidth=2, label="ECG Signal")
    
    # 🔥 Scatter heatmap (واضح جدًا)
    scatter = plt.scatter(
        np.arange(len(signal)),
        signal,
        c=cam_resized,
        cmap='jet',
        s=40,             # 👈 حجم أكبر للنقاط
        edgecolors='none'
    )
    
    plt.colorbar(scatter, label="Importance")
    
    plt.title("Grad-CAM Explanation (ECG)", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
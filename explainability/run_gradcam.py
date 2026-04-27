import numpy as np
from tensorflow.keras.models import load_model
from main import build_dataset
from explainability.gradcam import compute_gradcam, plot_gradcam


# تحميل الموديل
model = load_model("model/ecg_model.keras", compile=False)

# تحميل البيانات
X, y = build_dataset()

# اختيار عينة
sample_index = 0
signal = X[sample_index]

# prediction
pred = model.predict(signal[np.newaxis, ...])[0]

# أعلى class
class_idx = np.argmax(pred)

print("Predicted class:", class_idx)

# حساب Grad-CAM
cam = compute_gradcam(model, signal, class_idx)

# رسم
plot_gradcam(signal, cam)
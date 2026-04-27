import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import random
import time

from explainability.gradcam import compute_gradcam
from preprocessing.ecg_preprocessing import preprocess_ecg

# =========================
# Config
# =========================
st.set_page_config(page_title="ECG AI System", layout="wide")

# =========================
# Sidebar
# =========================
st.sidebar.title("🫀 ECG AI System")

st.sidebar.markdown("""
### About
Advanced ECG Analysis using:
- Deep Learning
- Multi-label Classification
- Explainable AI (Grad-CAM)
- Clinical Reasoning Engine

⚠️ Research use only
""")

# =========================
# Labels
# =========================
labels = [
    'SR', 'NORM', 'ABQRS', 'IMI', 'ASMI',
    'LVH', 'NDT', 'LAFB', 'AFIB', 'ISC_',
    'PVC', 'IRBBB', 'STD_'
]

# =========================
# Load model
# =========================
@st.cache_resource
def load_model_cached():
    return load_model("model/ecg_model.keras", compile=False)

model = load_model_cached()

# =========================
# ECG Plot
# =========================
def plot_ecg(signal, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color='cyan')))
    fig.update_layout(template="plotly_dark", height=300, title=title)
    return fig


# =========================
# 🔥 Loading ECG Animation
# =========================
def loading_animation():

    placeholder = st.empty()

    for i in range(20):
        fake_signal = np.sin(np.linspace(0, 3*np.pi, 200) + i*0.3)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=fake_signal, mode='lines', line=dict(color='lime')))

        fig.update_layout(
            template="plotly_dark",
            height=200,
            title="Analyzing ECG..."
        )

        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.05)

    placeholder.empty()
# =========================
# Grad-CAM Plot
# =========================
def plot_gradcam(signal, cam):

    cam_resized = np.interp(
        np.linspace(0, len(cam), num=len(signal)),
        np.arange(len(cam)),
        cam
    )

    important = cam_resized > 0.6

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color='white')))
    fig.add_trace(go.Scatter(
        y=signal[important],
        x=np.where(important)[0],
        mode='markers',
        marker=dict(color='red', size=8)
    ))

    fig.update_layout(template="plotly_dark", height=350, title="Grad-CAM")
    return fig


# =========================
# Quantitative XAI
# =========================
def analyze_cam(signal, cam):

    cam_resized = np.interp(
        np.linspace(0, len(cam), num=len(signal)),
        np.arange(len(cam)),
        cam
    )

    cam_resized /= (np.max(cam_resized) + 1e-8)

    important = cam_resized > 0.6
    ratio = np.sum(important) / len(signal)

    segments = [
        np.sum(important[:70]),
        np.sum(important[70:140]),
        np.sum(important[140:])
    ]

    region_names = ["early", "middle (QRS)", "late"]
    region = region_names[np.argmax(segments)]

    return ratio, region


# =========================
# ECG Features
# =========================
from scipy.signal import find_peaks

def extract_features(signal):

    peaks, _ = find_peaks(signal, distance=30)

    if len(peaks) > 1:
        rr = np.diff(peaks) / 100
        hr = 60 / np.mean(rr)
        rr_std = np.std(rr)
    else:
        hr = 0
        rr_std = 0

    return hr, rr_std

# =========================
# Clinical Reasoning
# =========================
def build_reasoning(pred, labels, hr, rr_std, ratio, region):

    top = np.argsort(pred)[::-1][:3]
    top_preds = [(labels[i], pred[i]) for i in top]

    text = "### 🧠 Clinical AI Report\n\n"

    text += "### Detected Conditions\n"
    for l, c in top_preds:
        text += f"- {l} ({c*100:.1f}%)\n"

    text += "\n"

    # =========================
    # Severity
    # =========================
    max_conf = top_preds[0][1]

    if max_conf > 0.85:
        severity = "Severe"
        risk = "High"
    elif max_conf > 0.65:
        severity = "Moderate"
        risk = "Medium"
    else:
        severity = "Mild"
        risk = "Low"

    text += f"### Severity: {severity}\n"
    text += f"### Risk: {risk}\n\n"

    # =========================
    # Physiology
    # =========================
    text += "### Physiological Analysis\n"
    text += f"- Heart Rate: {hr:.1f} bpm\n"
    text += f"- RR Variability: {rr_std:.3f}\n"

    if rr_std > 0.12:
        text += "- Irregular rhythm → possible AFIB\n"

    if hr > 100:
        text += "- Tachycardia detected\n"

    if hr < 60 and hr != 0:
        text += "- Bradycardia detected\n"

    text += "\n"

    # =========================
    # XAI
    # =========================
    text += "### Explainability\n"
    text += f"- Important signal ratio: {ratio*100:.1f}%\n"
    text += f"- Dominant region: {region}\n"

    if ratio < 0.15:
        text += "- Local spikes → ectopic beats\n"
    elif ratio < 0.35:
        text += "- QRS-related abnormalities\n"
    else:
        text += "- Global rhythm disturbance\n"

    text += "\n"

    # =========================
    # Recommendation
    # =========================
    text += "### Recommendation\n"

    if risk == "High":
        text += "⚠️ Immediate cardiology consultation required\n"
    elif risk == "Medium":
        text += "Follow-up ECG recommended\n"
    else:
        text += "Normal monitoring\n"

    return text

# =========================
# UI
# =========================
st.title("🫀 ECG AI Diagnosis Platform")
st.divider()

col1, col2, col3 = st.columns(3)

use_sample = col1.button("🔍 Test")
try_again = col2.button("🔁 Try Again")
uploaded = col3.file_uploader("Upload CSV", type=["csv"])

data = None

if use_sample or try_again:
    loading_animation()
    from main import build_dataset
    X, _ = build_dataset()
    data = X[random.randint(0, len(X)-1)].squeeze()

elif uploaded:
    data = np.loadtxt(uploaded, delimiter=",")

# =========================
# Results
# =========================
if data is not None:

    if len(data) != 200:
        st.error("ECG must be 200 values")

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_ecg(data, "Raw ECG"))

        processed = preprocess_ecg(data)

        with col2:
            st.plotly_chart(plot_ecg(processed, "Processed ECG"))

        signal = processed.reshape(200, 1)
        pred = model.predict(signal[np.newaxis, ...])[0]

        cam = compute_gradcam(model, signal, np.argmax(pred))
        ratio, region = analyze_cam(processed, cam)

        hr, rr_std = extract_features(processed)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📊 Predictions")
            for i in np.argsort(pred)[::-1][:3]:
                st.write(f"{labels[i]} — {pred[i]*100:.2f}%")

        with col4:
            report = build_reasoning(pred, labels, hr, rr_std, ratio, region)
            st.markdown(report)

        st.subheader("🔍 Grad-CAM")
        st.plotly_chart(plot_gradcam(processed, cam))    
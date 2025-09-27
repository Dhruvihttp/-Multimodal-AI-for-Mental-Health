import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Feature extraction: text (placeholder for real BERT or NLP!)
def extract_text_features(text):
    if not text: return np.zeros(768)
    # Placeholder: Replace with BERT or NLP model for production
    return np.random.normal(size=(768,))

# Audio feature extraction using MFCCs (through librosa)
def extract_audio_features(audio_bytes):
    if not audio_bytes: return np.zeros(13)
    try:
        import librosa
        import io
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    except Exception:
        return np.zeros(13)

# Flexible wearable/healthcare feature extraction for any CSV
def extract_flexible_wearable_features(file):
    try:
        df = pd.read_csv(file)
        features = []
        for col in df.columns:
            # Convert all values to numeric if possible
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            features.extend([
                numeric_col.mean(skipna=True),
                numeric_col.std(skipna=True),
                numeric_col.min(skipna=True),
                numeric_col.max(skipna=True)
            ])
        return np.array(features)
    except Exception:
        return np.zeros(4*len(pd.read_csv(file).columns))

# Demo fusion model for prediction (replace with a real model)
def fusion_model_predict(text_feat, audio_feat, wear_feat):
    feats = np.concatenate([text_feat, audio_feat, wear_feat])
    risk = float(np.clip(feats.sum() % 1, 0, 1))
    return risk

def plot_risk_bar(risk_score):
    fig, ax = plt.subplots(figsize=(5,1))
    ax.barh(['Risk Level'], [risk_score], color='crimson' if risk_score > 0.6 else 'orange' if risk_score > 0.4 else 'limegreen')
    ax.set_xlim(0, 1)
    ax.set_xlabel('0=Low, 1=High')
    ax.set_title('Predicted Risk')
    st.pyplot(fig)

# Streamlit UI
st.title("🧠 Multimodal AI for Mental Health")
st.markdown("""
Upload one or more of:
- **Text** (journal, message, social post)
- **Audio** (voice note, .wav)
- **Any healthcare CSV** (numeric columns: wearables, survey scores, etc.)

Risk is predicted from all available inputs!
""")

# Inputs
user_text = st.text_area("Text (journal/journal entry/social post):")
audio_file = st.file_uploader("Upload audio (.wav)", type="wav")
if audio_file:
    audio_bytes = audio_file.read()
else:
    audio_bytes = None
health_file = st.file_uploader("Upload ANY numeric healthcare CSV:")

if st.button("Predict Risk"):
    st.write("Extracting features...")
    tf = extract_text_features(user_text)
    af = extract_audio_features(audio_bytes)
    if health_file:
        wf = extract_flexible_wearable_features(health_file)
    else:
        wf = np.zeros(0)

    # Demo: see feature vector shapes
    with st.expander("Show extracted feature vector shapes"):
        st.write("Text:", tf.shape)
        st.write("Audio:", af.shape)
        st.write("Healthcare/CSV:", wf.shape)

    # Predict and display
    risk = fusion_model_predict(tf, af, wf)
    st.success(f"🩺 Predicted mental health risk score: **{risk:.2f}**")
    plot_risk_bar(risk)

    if risk > 0.7:
        st.error("🔴 High detected risk. Consider seeking supportive resources.")
    elif risk > 0.4:
        st.warning("🟠 Moderate risk. Practice self-care and check-in regularly.")
    else:
        st.success("🟢 Low risk. Keep up your healthy routines!")

    st.caption("This demo predicts risk using simulated logic. Connect your own models for production.")

st.markdown("""
---
**Sample CSV for upload (for testing this app):**
""")
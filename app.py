import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd
from sklearn.calibration import calibration_curve

# ============================================
# App Config
# ============================================

st.set_page_config(
    page_title="Automated Pneumonia Detection from Chest X-Ray Images",
    layout="centered"
)

st.title("🫁 Automated Pneumonia Detection from Chest X-Ray Images")
st.write(
    "Upload chest X-ray images to predict **Pneumonia** vs **Normal**, "
    "compare models, analyze confidence calibration, and perform batch inference."
)

# ============================================
# Sidebar Controls
# ============================================

st.sidebar.header("⚙️ App Settings")

# theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.01
)

show_gradcam = st.sidebar.checkbox("Show Grad-CAM Explainability", value=True)

enable_comparison = st.sidebar.checkbox("Enable Model Comparison", value=True)

# ============================================
# Theme Styling (Safe CSS)
# ============================================

# if theme == "Dark":
#     card_bg = "#1e1e1e"
#     card_border = "#444"
# else:
#     card_bg = "#f9f9f9"
#     card_border = "#ddd"


# ============================================
# Load Models
# ============================================

@st.cache_resource
def load_models():
    densenet = tf.keras.models.load_model("model/densenet_pneumonia_model.keras")
    resnet = tf.keras.models.load_model("model/resnet_pneumonia_model.keras")
    return densenet, resnet

with st.spinner("Loading models..."):
    densenet_model, resnet_model = load_models()

st.success("DenseNet121 & ResNet50 loaded successfully!")

# ============================================
# Helpers
# ============================================

def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_with_model(model, img_array, threshold):
    prob = model.predict(img_array)[0][0]
    if prob >= threshold:
        return "Pneumonia", prob
    else:
        return "Normal", 1 - prob

def generate_gradcam(model, img_array, last_conv_layer):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if isinstance(preds, list):
            preds = preds[0]
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return cv2.resize(heatmap, (224, 224))

# ============================================
# Single Image Inference
# ============================================

st.subheader("📤 Single Image Prediction")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", width="content")
    img_array = preprocess_image(image)

    col1, col2 = st.columns(2)

    # DenseNet
    with col1:
        label_dn, conf_dn = predict_with_model(
            densenet_model, img_array, confidence_threshold
        )
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### DenseNet121")
        st.metric("Prediction", label_dn)
        st.metric("Confidence", f"{conf_dn*100:.2f}%")

        if show_gradcam:
            heatmap = generate_gradcam(
                densenet_model, img_array, "conv5_block16_concat"
            )
            fig, ax = plt.subplots()
            ax.imshow(image.resize((224, 224)))
            ax.imshow(heatmap, cmap="jet", alpha=0.4)
            ax.axis("off")
            st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # ResNet
    if enable_comparison:
        with col2:
            label_rs, conf_rs = predict_with_model(
                resnet_model, img_array, confidence_threshold
            )
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### ResNet50")
            st.metric("Prediction", label_rs)
            st.metric("Confidence", f"{conf_rs*100:.2f}%")

            if show_gradcam:
                heatmap = generate_gradcam(
                    resnet_model, img_array, "conv5_block3_out"
                )
                fig, ax = plt.subplots()
                ax.imshow(image.resize((224, 224)))
                ax.imshow(heatmap, cmap="jet", alpha=0.4)
                ax.axis("off")
                st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Batch Upload & Inference (DenseNet + ResNet)
# ============================================

st.subheader("📂 Batch Upload (Multiple X-Rays)")

batch_files = st.file_uploader(
    "Upload multiple chest X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if batch_files:
    results = []

    with st.spinner("Running batch predictions (DenseNet121 & ResNet50)..."):
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            img_arr = preprocess_image(img)

            # DenseNet prediction
            label_dn, conf_dn = predict_with_model(
                densenet_model, img_arr, confidence_threshold
            )

            # ResNet prediction
            label_rs, conf_rs = predict_with_model(
                resnet_model, img_arr, confidence_threshold
            )

            results.append({
                "Filename": file.name,
                "DenseNet121 Prediction": label_dn,
                "DenseNet121 Confidence (%)": round(conf_dn * 100, 2),
                "ResNet50 Prediction": label_rs,
                "ResNet50 Confidence (%)": round(conf_rs * 100, 2),
            })

    # ----------------------------------------
    # Build comparison table
    # ----------------------------------------

    df = pd.DataFrame(results)

    # Agreement flag
    df["Agreement"] = np.where(
        df["DenseNet121 Prediction"] == df["ResNet50 Prediction"],
        "Yes",
        "No"
    )

    # Confidence difference
    df["Confidence Δ (%)"] = (
        df["DenseNet121 Confidence (%)"]
        - df["ResNet50 Confidence (%)"]
    ).abs().round(2)

    # User-friendly index (start from 1)
    df.index = df.index + 1

    # ----------------------------------------
    # Styling for readability
    # ----------------------------------------

    def highlight_prediction(val):
        if val == "Pneumonia":
            return "background-color: #ffcccc; color: black;"
        return "background-color: #ccffcc; color: black;"

    def highlight_agreement(val):
        if val == "No":
            return "background-color: #fff3cd; color: black;"
        return ""

    styled_df = (
        df.style
        .applymap(
            highlight_prediction,
            subset=[
                "DenseNet121 Prediction",
                "ResNet50 Prediction"
            ]
        )
        .applymap(
            highlight_agreement,
            subset=["Agreement"]
        )
        .format({
            "DenseNet121 Confidence (%)": "{:.2f}",
            "ResNet50 Confidence (%)": "{:.2f}",
            "Confidence Δ (%)": "{:.2f}"
        })
    )

    # ----------------------------------------
    # Display table
    # ----------------------------------------

    st.subheader("📊 Batch Prediction Summary (Model Comparison)")

    st.dataframe(
        styled_df,
        use_container_width=True
    )

    # ----------------------------------------
    # Download CSV (clean, no styling)
    # ----------------------------------------

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Batch Results (CSV)",
        csv,
        "batch_predictions.csv",
        "text/csv"
    )



# ============================================
# Disclaimer
# ============================================

st.markdown("---")
st.warning(
    "⚠️ **Disclaimer:** This application is for educational and research purposes "
    "only and should not be used as a substitute for professional medical diagnosis."
)

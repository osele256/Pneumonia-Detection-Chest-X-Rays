import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pydicom
import gdown
import os
from PIL import Image

# Google Drive Model Link (update if needed)
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?id=11fyMiUohMB4zz6Mo822uFSe_4mCsf8js"
MODEL_PATH = "final_pneumonia_model.keras"

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

def download_model():
    """Download model from Google Drive if not available locally."""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        try:
            gdown.download(GOOGLE_DRIVE_LINK, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None
    return MODEL_PATH

@st.cache_resource
def load_pneumonia_model():
    """Load the model, downloading if needed."""
    model_path = download_model()
    if model_path:
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

def read_dicom_image(file, target_size=(224, 224)):
    """Read and process an image (DICOM or regular formats)."""
    try:
        if file.name.lower().endswith('.dcm'):
            dicom = pydicom.dcmread(file)
            if dicom.pixel_array is None:
                raise ValueError("DICOM file does not contain pixel data.")
            data = dicom.pixel_array.astype(np.float32)
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
            data = np.stack([data] * 3, axis=-1)
        else:
            img = Image.open(file).convert("RGB")
            data = np.array(img) / 255.0

        data = tf.image.resize(data, target_size).numpy()
        return data
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

def predict_pneumonia(model, img):
    """Run pneumonia prediction on an image."""
    if img is None or model is None:
        return None
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img_batch)[0][0]
    return {
        'probability': float(prediction),
        'prediction': 'Pneumonia' if prediction > 0.5 else 'Normal',
        'image': img
    }

def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/lungs.png", width=100)
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses deep learning to detect pneumonia from chest X-ray images. "
        "Upload an image to get a prediction."
    )
    
    st.sidebar.title("Supported Formats")
    st.sidebar.write("DICOM (.dcm), JPEG, PNG")

    # Main content
    st.title("Pneumonia Detection from Chest X-rays")
    st.write(
        "This app uses a ResNet50-based deep learning model to predict pneumonia presence in chest X-ray images."
    )
    
    uploaded_file = st.file_uploader("Upload an X-ray image...", type=["dcm", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            img = read_dicom_image(uploaded_file)

            if img is not None:
                st.image(img, caption="Uploaded X-ray", use_column_width=True)
            else:
                st.error("Failed to process the image. Try another file.")
                return

        if st.button("Run Pneumonia Detection"):
            with st.spinner("Analyzing image..."):
                model = load_pneumonia_model()
                if model is None:
                    st.error("Failed to load the model.")
                    return
                
                result = predict_pneumonia(model, img)

                with col2:
                    st.subheader("Analysis Results")

                    if result['prediction'] == 'Pneumonia':
                        st.error(f"**Prediction: {result['prediction']}**")
                    else:
                        st.success(f"**Prediction: {result['prediction']}**")
                    
                    st.metric(label="Pneumonia Probability", value=f"{result['probability']:.2%}")
                    st.progress(result['probability'])

                    if result['prediction'] == 'Pneumonia':
                        st.info("The model detected signs of pneumonia. Confirm with a healthcare professional.")
                    else:
                        st.info("No significant signs of pneumonia detected. Confirm with a healthcare professional.")

    st.markdown("---")
    st.caption("**Disclaimer:** This app is for educational purposes only and is not a substitute for medical advice.")

if __name__ == "__main__":
    main()

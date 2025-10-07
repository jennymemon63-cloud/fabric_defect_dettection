import os
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------- CONFIG ----------------------
FILE_ID = "1OnlE8hB07Dtn5ZE1VUU6oD-2UwZ2aAD0"  # your Google Drive file ID
MODEL_PATH = "best.pt"
# ----------------------------------------------------

# Function: download YOLO model from Google Drive if not found
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Downloading model from Google Drive... Please wait...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        else:
            st.error(f"❌ Failed to download model. HTTP Status: {response.status_code}")


# Function: Run inference on image
def inference_images(uploaded_file, model):
    image = Image.open(uploaded_file)
    st.image(image, caption="🧵 Uploaded Image", use_column_width=True)

    st.info("🔍 Running fabric defect detection...")
    results = model.predict(image)
    plotted = results[0].plot()[:, :, ::-1]

    if len(results[0].boxes) == 0:
        st.warning("⚠️ No defects detected.")
    else:
        st.success("✅ Defects detected successfully!")

    st.image(plotted, caption="Detected Defects", use_column_width=True)


# ---------------------- MAIN APP ----------------------
def main():
    st.title("🧵 Fabric Defect Detection using YOLOv8")

    # Step 1: Download model if needed
    download_model()

    # Step 2: Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        return

    # Sidebar info
    st.sidebar.header("📁 Options")
    st.sidebar.write("Upload a fabric image to detect defects.")

    # Step 3: Upload image
    uploaded_file = st.file_uploader("📸 Upload Fabric Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        inference_images(uploaded_file, model)


# Entry point
if __name__ == "__main__":
    main()

#os.environ["MPLCONFIGDIR"] = "/tmp"
#os.environ["OPENCV_OPENGL"] = "false"
#os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
#os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "0"
#os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
# Force OpenCV to use headless mode
#os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
#import cv2

#import os
#import numpy as np
#from PIL import Image
#import streamlit as st
#from ultralytics import YOLO

#MODEL_DIR = './runs/fabric_defect/yolov8s_fd/weights/best.pt'
import os
import requests
import streamlit as st
from ultralytics import YOLO

# Google Drive file ID (tumhare link se nikala gaya)
FILE_ID = "1OnlE8hB07Dtn5ZE1VUU6oD-2UwZ2aAD0"
MODEL_PATH = "best.pt"

# Function: download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Downloading model from Google Drive... Please wait.")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        else:
            st.error(f"❌ Failed to download model. HTTP Status: {response.status_code}")

# Main Streamlit app
def main():
    st.title("🧵 Fabric Defect Detection (YOLOv8)")

    # Step 1: Download model if needed
    download_model()

    # Step 2: Load model
    try:
        model = YOLO(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        return

    # Step 3: Upload image for inference
    uploaded_file = st.file_uploader("📸 Upload fabric image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_path = "uploaded_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())
        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        # Run prediction
        results = model(img_path)
        results.save("result.jpg")
        st.image("result.jpg", caption="Detected Defects", use_column_width=True)

if __name__ == "__main__":
    main()


def main():
    # load a model
    model = YOLO(MODEL_DIR)

    st.sidebar.header("**Fabric Defect Detection**")

    for animal in sorted(os.listdir('./data/raw')):
       st.sidebar.markdown(f"- *{animal.capitalize()}*")

    st.title("Real-time Fabric Defect Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time Fabric Defect detection.")

    # Load image or video
   uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    #if uploaded_file:
       if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file, model)
        
        if uploaded_file.type.startswith('video'):
            inference_video(uploaded_file)


def inference_images(uploaded_file, model):
    image = Image.open(uploaded_file)
     predict the image
    predict = model.predict(image)

    # plot boxes
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # open the image.
    st.image(plotted, caption="Detected Image", width=600)
    #logging.info("Detected Image")


if __name__=='__main__':

    main()









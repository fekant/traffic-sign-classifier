import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

# ---------------------- CONFIGURATION ----------------------
MODEL_PATH = "traffic_classifier.h5"
META_PATH = "Meta"
IMG_SIZE = (30, 30)

# ---------------------- CLASS LABELS ----------------------
classes = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing veh over 3.5 tons", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles", 16: "Veh > 3.5 tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve left", 20: "Dangerous curve right",
    21: "Double curve", 22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End speed + passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left", 38: "Keep right",
    39: "Keep left", 40: "Roundabout mandatory", 41: "End of no passing",
    42: "End no passing vehicle > 3.5 tons"
}

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_cnn_model():
    return load_model("traffic_classifier.h5")

model = load_cnn_model()

# ---------------------- IMAGE CLASSIFICATION ----------------------
def classify_image(img):
    image = img.resize(IMG_SIZE)
    image = np.expand_dims(np.array(image), axis=0)
    pred = model.predict(image)
    pred_class = np.argmax(pred, axis=1)[0]
    return pred_class, classes.get(pred_class, "Unknown")

# ---------------------- STREAMLIT UI ----------------------
st.title("🚦 Traffic Sign Classifier")
st.write("Upload an image of a traffic sign to identify its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("\n")

    if st.button("Classify"):
        pred_class, label = classify_image(img)
        st.success(f"Prediction: {label} (Class ID: {pred_class})")

        # Display reference image if available
        meta_img_path = os.path.join(META_PATH, f"{pred_class}.png")
        if os.path.exists(meta_img_path):
            st.image(meta_img_path, caption=f"Reference: {label}", use_column_width=False)

st.markdown("---")
st.caption("Developed for Streamlit integration. Extendable with pothole detection models.")

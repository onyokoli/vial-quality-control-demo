#app.py

import streamlit as st
import tensorflow as tf
from PIL import Image # Pillow for image loading
import numpy as np
import os

# Page Configuration 
st.set_page_config(layout="wide")

# Global Configuration 
MODEL_NAME = 'vial_classifier.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CLASS_NAMES = ['good', 'bad'] # Make sure this matches the order used during training!

# Load the Trained Model

@st.cache_resource
def load_keras_model(model_path):
    """Loads the saved Keras model."""
    # Check if model file exists right before loading
    if not os.path.exists(model_path):
         # Use st.error inside the function if needed, but printing is safer here
         print(f"ERROR: Model file not found at {model_path}")
         st.error(f"Fatal Error: Model file not found at {model_path}. Please ensure '{MODEL_NAME}' is in the same directory as app.py.")
         # Return None or raise an exception if the file isn't found
         # Returning None is handled later in the UI section
         return None
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!") # Console print is fine here
        return model
    except Exception as e:
        # Provide more specific error feedback in the app and console
        st.error(f"Error loading model: {e}")
        print(f"Error loading model from {model_path}: {e}")
        return None

# Construct the path to the model file 
model_path = os.path.join(os.path.dirname(__file__), MODEL_NAME)
# Load the model (function call is fine here now that set_page_config is first)
model = load_keras_model(model_path)

# Preprocessing Function 
def preprocess_image(image):
    """Preprocesses the uploaded image to match model input requirements."""
    # Resize image
    image = image.resize(IMG_SIZE)
    # Convert image to numpy array
    image_array = np.array(image)
    # Ensure image is RGB 
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0) # Shape becomes (1, H, W, 3)
    # Apply MobileNetV2 preprocessing
    preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return preprocessed_image

# Streamlit App UI
st.title("🧪 Vial Quality Control Demo")
st.write("Upload an image of a vial to classify it as 'good' or 'bad' using a pre-trained Deep Learning model (MobileNetV2).")
st.write("---")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Vial Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])

    if uploaded_file is not None:
        # Display the uploaded image
        try:
            image = Image.open(uploaded_file).convert('RGB') # Ensure image is RGB
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        except Exception as e:
            st.error(f"Error opening or displaying image: {e}")
            uploaded_file = None
    else:
        image = None


with col2:
    st.header("Prediction Result")
    if uploaded_file is not None and model is not None and image is not None:
        with st.spinner('Analyzing image...'):
            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Make prediction
            try:
                prediction = model.predict(preprocessed_image)
                probability_bad = prediction[0][0] # Get the probability of class 'bad'

                # Determine predicted class
                if probability_bad > 0.5:
                    predicted_class_index = 1 # bad
                else:
                    predicted_class_index = 0 # good

                predicted_class_name = CLASS_NAMES[predicted_class_index]
                # Calculate confidence for the predicted class
                confidence = probability_bad if predicted_class_index == 1 else 1 - probability_bad

                # Display results
                st.write("") 
                st.write("##### Analysis Complete:")

                if predicted_class_name == 'bad':
                    st.error(f"Predicted Class: **BAD**")
                else:
                    st.success(f"Predicted Class: **GOOD**")

                st.write(f"Confidence: **{confidence:.2%}**")
                st.progress(float(confidence)) # Show confidence bar
                st.caption(f"Model's raw probability score for 'bad': {probability_bad:.4f}")

            except Exception as e:
                 st.error(f"Error during prediction: {e}")


    elif model is None:
         st.warning("Model could not be loaded. Cannot perform analysis.")
    else:
        st.info("Awaiting image upload...")

st.write("---")
st.caption("Model based on MobileNetV2 | MVTec AD Vial Dataset | Demo by Onyinye Okoli") 

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2  # Import OpenCV for image operations
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = tf.keras.models.load_model('m.h5')  # Replace with the actual path to your trained model

# Streamlit app
st.title("Digit Recognition App")

uploaded_image = st.file_uploader("Upload a Grayscale Image", type=["jpg", "png", "jpeg"])

# Create a canvas for drawing
canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image):
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        grayscale_image = image

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(grayscale_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize the image
    normalized_image = resized_image / 255.0
    
    # Reshape the image to match the model's input shape (-1, 28, 28, 1)
    input_image = normalized_image.reshape((1, 28, 28, 1))
    
    return input_image

def predict_digit(image_data):
    # Preprocess the image
    input_image = preprocess_image(image_data)

    # Check if the input image contains a valid digit
    is_valid_digit = is_valid_input(input_image)

    if is_valid_digit:
        # Make predictions
        prediction = model.predict(input_image)
        predicted_digit = np.argmax(prediction)

        return predicted_digit, prediction
    else:
        return None, None

# Define a function to check if the input is a valid digit (you can customize this)
def is_valid_input(input_image):
    # Implement your logic to determine if the input image is a valid digit
    # For example, you can check for certain characteristics or thresholds
    return True  # Change this condition based on your criteria

# Streamlit button for prediction
if st.button("Predict"):
    if canvas.image_data is not None:
        # Predict the digit and get the prediction result for drawn image
        predicted_digit, prediction = predict_digit(canvas.image_data)
        
        if predicted_digit is not None:
            st.image(canvas.image_data, caption=f"Predicted Digit: {predicted_digit}", use_column_width=True)
            st.write("Prediction Probabilities:")
            for i in range(10):
                st.write(f"Digit {i}: {prediction[0][i]:.4f}")
        else:
            st.write("Input is not a valid digit. Please draw a digit between 0 and 9.")
    elif uploaded_image is not None:
        # Predict the digit and get the prediction result for uploaded image
        pil_image = Image.open(uploaded_image)
        pil_image = pil_image.convert("L")  # Convert to grayscale
        uploaded_image_array = np.array(pil_image)
        predicted_digit, prediction = predict_digit(uploaded_image_array)

        if predicted_digit is not None:
            st.image(uploaded_image, caption=f"Predicted Digit: {predicted_digit}", use_column_width=True)
            st.write("Prediction Probabilities:")
            for i in range(10):
                st.write(f"Digit {i}: {prediction[0][i]:.4f}")
        else:
            st.write("Input is not a valid digit. Please upload a grayscale image.")

# Add a sidebar with some information
st.sidebar.title("About")
st.sidebar.info("This is a simple digit recognition app using TensorFlow and Streamlit.")

# Add a footer
st.markdown("---")
st.markdown("By Your Name")

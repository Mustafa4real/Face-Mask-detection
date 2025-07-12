import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load the model
model = load_model("trained_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(maxsplit=1)[1] for line in f.readlines()]

# Function to preprocess and predict the image
def predict_image_class(img_path):
    # Load image and resize to model input size (assumed 224x224)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0,1]

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    print(f"Predicted class: {labels[class_idx]} (Confidence: {confidence:.2f})")

# Example usage (wasn't used in the training)
predict_image_class("test_image.png")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


try:
    model = tf.keras.models.load_model("WeedDetectionModel.h5") #greenintel testing 10 epoch
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def test_single_image(image_path):
    """ Function to classify a single image using the trained model """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image could not be loaded. Check the path: {image_path}")
        return

    img = cv2.resize(img, (224, 224))  
    img = np.array(img, dtype=np.float32)  
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  

    prediction = model.predict(img)
    detected_class = np.argmax(prediction[0])

    print(f"Detected Class: {'Weed' if detected_class == 1 else 'No Weed'}")

image_path = r"C:\Users\Administrator\Downloads\TestImages\randomplant.jpg"
test_single_image(image_path)
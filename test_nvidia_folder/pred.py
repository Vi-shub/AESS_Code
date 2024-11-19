import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved MobileNetV1 model
model_path = 'Mobilenet_updated.h5'  # Replace with the path to your saved model file
loaded_model = load_model(model_path)

# Function to preprocess input image for prediction
def preprocess_input_image(image_path):
    img = image.load_img(image_path, target_size=(180, 180))  # Adjust the target size as per your model's input shape
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the input image (if your model requires this)
    return img

# Function to make predictions
def make_predictions(image_paths, model):
    predictions_list = []
    for image_path in image_paths:
        input_image = preprocess_input_image(image_path)
        predictions = model.predict(input_image)
        predictions_list.append(predictions)
    return predictions_list

# Directory containing the images you want to make predictions on
image_directory = './output_folder'  # Replace with the path to your image folder

# List of image file names in the directory
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png','JPG'))]

# Make predictions
predictions = make_predictions(image_files, loaded_model)
print(len(predictions))
# Your custom model is already trained, so no decoding is necessary if it's a regression model or a specific task.
# If it's a classification model, you might want to map prediction indices to class labels as needed.
# print(type(predictions))
count = 0
for i, prediction in enumerate(predictions):
    output = 'Not Healthy' if np.argmax(prediction)==1 else 'Healthy'
    if output=='Healthy':
        count += 1
    #print(f"Predictions for {image_files[i]}:",output)
print(count/len(predictions))

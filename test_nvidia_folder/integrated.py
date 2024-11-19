import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os 
from PIL import Image
from io import BytesIO

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
image_directory = r'/home/jetson/Downloads/test_nvidia_folder/output_folder'  # Replace with the path to your image folder

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

def pre_processing(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Median Filtering
    noisy_image = cv2.medianBlur(img, 5)

     # CLAHE
    cl = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2Lab)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
    cl[:, :, 0] = clahe.apply(cl[:, :, 0])
    cl = cv2.cvtColor(cl, cv2.COLOR_Lab2RGB)

    # Sharpening
    sharpen_filter_mild = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_image = cv2.filter2D(cl, -1, sharpen_filter_mild)

    return sharp_image


def leaf_isolation(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_range = np.array([30, 50, 50])
    upper_range = np.array([60, 250, 250])
    mask = cv2.inRange(hsv_image, lower_range, upper_range)
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 70000
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    combined_leaves = np.zeros_like(image)
    leaf_count = 1
    for contour in filtered_contours:
        leaf_mask = np.zeros_like(closed_mask)
        cv2.drawContours(leaf_mask, [contour], -1, 255, thickness=cv2.FILLED)
        isolated_leaf = cv2.bitwise_and(image, image, mask=leaf_mask)
        combined_leaves = cv2.add(combined_leaves, isolated_leaf)
        leaf_count += 1
    return combined_leaves

def process_image_with_CIVE(image_path):
    def get_CIVE_band(img):
        img = cv2.GaussianBlur(img, (35, 35), 0)
        CIVE_band = 0.441 * img[:, :, 0] - 0.881 * img[:, :, 1] + 0.385 * img[:, :, 2] + 18.787
        normalized_CIVE_band = (((CIVE_band + abs(CIVE_band.min())) / CIVE_band.max())).astype(np.uint8)
        return normalized_CIVE_band

    def apply_CIVE_mask(img, vegetation_index_band):
        ret, otsu = cv2.threshold(vegetation_index_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masked_img = cv2.bitwise_and(img, img, mask=otsu)
        return masked_img

    img = np.array(Image.open(image_path))

    CIVE_band = get_CIVE_band(img)
    CIVE_masked = apply_CIVE_mask(img, CIVE_band)

    return CIVE_masked


def segmentation(leaves_image, output_dir):
    lower_green = (0, 100, 0)
    upper_green = (50, 255, 50)
    mask = cv2.inRange(leaves_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    os.makedirs(output_dir, exist_ok=True)
    margin = 100
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, leaves_image.shape[1] - x)
        h = min(h + 2 * margin, leaves_image.shape[0] - y)
        roi = leaves_image[y:y+h, x:x+w]
        new_size = (int(0.5 * roi.shape[1]), int(0.5 * roi.shape[0]))
        roi = cv2.resize(roi, new_size)
        output_path = os.path.join(output_dir, f'leaf_{i}.jpg')
        cv2.imwrite(output_path, roi)

    print(f'{len(contours)} leaves saved in {output_dir}')

# Access the webcam
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Adjust parameters as needed

# Record start time
start_time = cv2.getTickCount()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Preprocess the frame for prediction
    input_image = preprocess_input_image(frame)

    # Make predictions
    predictions = loaded_model.predict(input_image)
    output = 'Not Healthy' if np.argmax(predictions[0]) == 1 else 'Healthy'

    # Display the output on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, output, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Webcam', frame)

    # Write the frame to the output video
    out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Record end time
end_time = cv2.getTickCount()

# Calculate elapsed time
elapsed_time = (end_time - start_time) / cv2.getTickFrequency()

# Release the video writer, webcam, and close all windows
out.release()
cap.release()
cv2.destroyAllWindows()

# Directory containing the images you want to make predictions on
image_directory = r'/home/jetson/Downloads/test_nvidia_folder/output_folder'  # Replace with the path to your image folder

# List of image file names in the directory
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png','JPG'))]

# Make predictions
predictions = make_predictions(image_files, loaded_model)
print(len(predictions))
count = 0
for i, prediction in enumerate(predictions):
    output = 'Not Healthy' if np.argmax(prediction)==1 else 'Healthy'
    if output=='Healthy':
        count += 1
print(count/len(predictions))


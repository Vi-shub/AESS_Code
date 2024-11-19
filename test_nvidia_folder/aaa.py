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
import datetime
import shutil

test_img_path = r"C:\Users\smsha\Desktop\Jetson\test_nvidia_folder/healthy3.jpg"
test_img = cv2.imread(test_img_path)

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
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv color space", hsv_image)
    # Define color range for leaves
    lower_range = np.array([30, 50, 50])
    upper_range = np.array([60, 250, 250])

    # Create a binary mask based on the color range
    mask = cv2.inRange(hsv_image, lower_range, upper_range)

    # Apply morphological operations to refine the mask
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("binary mask", closed_mask)
    # Find contours in the refined mask
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and shape
    min_contour_area = 70000
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    # Create an empty image to combine isolated leaves
    combined_leaves = np.zeros_like(image)

    leaf_count = 1
    for contour in filtered_contours:
        # Create a mask for the current leaf
        leaf_mask = np.zeros_like(closed_mask)
        cv2.drawContours(leaf_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original image to isolate the leaf
        isolated_leaf = cv2.bitwise_and(image, image, mask=leaf_mask)
        cv2.imshow(f"isolated_leaf_{leaf_count}", isolated_leaf)

        # Add the isolated leaf to the combined_leaves image
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
    # Define the color range for green (in BGR format)
    lower_green = (0, 100, 0)
    upper_green = (100, 255, 100)

    # Create a mask for the green region
    mask = cv2.inRange(leaves_image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a directory to save the new images
    os.makedirs(output_dir, exist_ok=True)
    min_contour_area = 2
    # Margin to add around the leaf (increase for more zoomed-out view)
    margin = 100

    valid_contours = []  # List to store valid contours

    for i, contour in enumerate(contours):
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Add margin around the bounding box
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, leaves_image.shape[1] - x)
        h = min(h + 2 * margin, leaves_image.shape[0] - y)

        # Additional constraint: Filter contours based on area
        contour_area = cv2.contourArea(contour)
        if contour_area < min_contour_area:
            print(f"Contour {i} skipped due to small area: {contour_area}")
            continue  # Skip contours with small areas

        valid_contours.append(contour)

        # Extract the region of interest (ROI)
        roi = leaves_image[y:y+h, x:x+w]

        # Resize the ROI to make it smaller (adjust as needed)
        new_size = (max(int(0.5 * roi.shape[1]), 1), max(int(0.5 * roi.shape[0]), 1))  # Ensure the size is at least 1x1
        roi = cv2.resize(roi, new_size)

        # Save the ROI to the output folder
        output_path = os.path.join(output_dir, f'leaf_{i}.jpg')
        cv2.imwrite(output_path, roi)

    print(f'{len(valid_contours)} valid leaves saved in {output_dir}')




cnn = load_model('test_nvidia_folder\model_plant_detection.h5')
start_time = datetime.datetime.now()
test_image = image.load_img(test_img_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = test_image / 255
test_image = np.expand_dims(test_image, axis=0)

output_dir = r"C:\Users\smsha\Desktop\Jetson\test_nvidia_folder\output_folder"

# Perform classification
result = cnn.predict(test_image)



if result[0] < 0:
    print("It's is not a plant")
else:
    test_image_cv = cv2.imread(test_img_path)
    # print("It's a plant")
    # print(result)
    # pre_processed_image = pre_processing(test_img_path)

    # cv2.imshow("preprocessed image", pre_processed_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    isolated_leaf_image =process_image_with_CIVE(test_img_path)
    # cv2.imshow("isolated_leaf_image", isolated_leaf_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows
    segmentation(isolated_leaf_image, output_dir)
    
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime
# Load the saved MobileNetV1 model
model_path = 'test_nvidia_folder\Mobilenet_updated.h5'  # Replace with the path to your saved model file
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
image_directory = r'C:\Users\smsha\Desktop\Jetson\test_nvidia_folder\output_folder'  # Replace with the path to your image folder

# List of image file names in the directory
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png','JPG'))]

# Make predictions
predictions = make_predictions(image_files, loaded_model)
print(len(predictions))
end_time = datetime.now()
# Your custom model is already trained, so no decoding is necessary if it's a regression model or a specific task.
# If it's a classification model, you might want to map prediction indices to class labels as needed.
# print(type(predictions))
count = 0

for i, prediction in enumerate(predictions):
    output = 'Not Healthy' if np.argmax(prediction)==1 else 'Healthy'
    print(i)
    if output=='Healthy':
        count += 1
    #print(f"Predictions for {image_files[i]}:",output)

print(end_time - start_time)
not_healthy = round((1-count/len(predictions))*100,2)
print(f"{not_healthy}% plant is not healthy")

#shutil.rmtree(r'C:\Users\smsha\Desktop\Jetson\test_nvidia_folder\output_folder')





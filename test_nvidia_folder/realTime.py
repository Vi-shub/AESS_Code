import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import datetime
# Load the saved MobileNetV1 model
model_path = 'Mobilenet_updated.h5'  # Replace with the path to your saved model file
loaded_model = load_model(model_path)

# Function to preprocess input image for prediction
def preprocess_input_image(img):
    img = cv2.resize(img, (180, 180))  # Adjust the target size as per your model's input shape
    img = img / 255.0  # Normalize the input image (if your model requires this)
    return np.expand_dims(img, axis=0)

# Access the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret: break
    start_time = datetime.datetime.now()

    # Preprocess the frame for prediction
    input_image = preprocess_input_image(frame)

    # Make predictions
    predictions = loaded_model.predict(input_image)
    
    output = 'Not Healthy' if np.argmax(predictions[0]) == 1 else 'Healthy'
    end = datetime.datetime.now()
    
    # Display the output
    font = cv2.FONT_HERSHEY_SIMPLEX
    if output == 'Not Healthy':
    	cv2.putText(frame, output, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
    	cv2.putText(frame, output, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    #cv2.putText(frame, str(end - start_time), (10, 90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Webcam', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

def calculate_mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

def extract_frame(video_path, timestamp_ms, mse_threshold=1000):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Set the position of the video to the given timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    
    # Read the frame at the given timestamp
    ret, frame = cap.read()
    
    # Check if the frame is read successfully
    if not ret:
        print("Error: Could not read frame at the given timestamp.")
        return

    # Check MSE with the previous frame
    if 'previous_frame' in locals():
        mse = calculate_mse(previous_frame, frame)
        print(f"MSE with previous frame at timestamp {timestamp_ms}: {mse}")
        if mse < mse_threshold:
            print(f"Skipping frame at timestamp {timestamp_ms} due to low MSE ({mse}) with the previous frame.")
            cap.release()
            return
    
    # Save the extracted frame
    cv2.imwrite(f"C:/Users/smsha/Desktop/Jetson/test_nvidia_folder/extract/extracted_frame{timestamp_ms}.jpg", frame)
    
    # Store the current frame for the next iteration
    previous_frame = frame.copy()
    
    # Release the video capture object
    cap.release()

# Example usage
video_path = "C:/Users/smsha/Desktop/Jetson/test_nvidia_folder/input.mp4"

beyond = 33000
end = 43000
interval = 100
for i in range(beyond, end, interval):
    extract_frame(video_path, i)

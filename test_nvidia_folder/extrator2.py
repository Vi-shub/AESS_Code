import cv2
import numpy as np

def extract_frame(video_path, timestamp_ms, correlation_threshold=0.6,prev_frame=None):
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

    # Check correlation with the previous frame
    if prev_frame is not None:
        print("here")
        correlation = cv2.matchTemplate(prev_frame, frame, cv2.TM_CCORR_NORMED)[0][0]
        if correlation > correlation_threshold:
            print(f"Skipping frame at timestamp {timestamp_ms} due to high correlation ({correlation}) with the previous frame.")
            cap.release()
            return frame
    else:
        cv2.imwrite("C:/Users/smsha/Desktop/Jetson/test_nvidia_folder/extract/extracted_frame"+str(timestamp_ms)+".jpg", frame)
        cap.release()
        return frame



        
    
    # Save the extracted frame
    cv2.imwrite("C:/Users/smsha/Desktop/Jetson/test_nvidia_folder/extract/extracted_frame"+str(timestamp_ms)+".jpg", frame)

    
    # Release the video capture object
    cap.release()
    return frame

# Example usage
video_path = "C:/Users/smsha/Desktop/Jetson/test_nvidia_folder/input.mp4"

beyond = 33000
end = 43000
interval = 100
prev_frame = None
for i in range(beyond, end, interval):
    prev_frame=extract_frame(video_path, i,prev_frame)

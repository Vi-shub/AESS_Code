import cv2
import os

def extract_frames(video_filename, output_folder, interval_ms):
    # Open the video file
    cap = cv2.VideoCapture(video_filename)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_filename}'")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video information
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {total_frames}")

    # Calculate the frame interval based on the desired time interval
    frame_interval = int(interval_ms / 1000 * fps)

    # Loop through each frame and save it as an image at the specified interval
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frames extraction complete.")
            break

        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_filename, frame)

        # Display progress
        print(f"Extracted frame {frame_number + 1}/{total_frames}")

        # Move to the next frame at the specified interval
        frame_number += frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Specify the video file name (assuming it's in the same directory as the script)
    video_filename = "C:/Users/smsha/Desktop/Jetson/test_nvidia_folder/input3.mp4"

    # Specify the output folder for extracted frames
    output_folder = "C:/Users/smsha/Desktop/Jetson/test_nvidia_folder/output_frames"

    # Specify the frame capture interval in milliseconds (5000ms = 5 seconds)
    interval_ms = 1000

    extract_frames(video_filename, output_folder, interval_ms)

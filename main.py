import cv2
import numpy as np
import os
from datetime import datetime
import time
import csv
import argparse
import json
from tabulate import tabulate  # For pretty-printing the CSV file

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def print_statistics(csv_file):
    """Print the CSV file in an organized table."""
    if not os.path.exists(csv_file):
        print("No statistics available. CSV file does not exist.")
        return

    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header row
        data = list(reader)     # Read the rest of the data

    # Print the data in a table format
    print(tabulate(data, headers=headers, tablefmt="pretty"))

def main(config, show_feed):
    # Load configuration parameters
    model_weights = config["model_weights"]
    model_config = config["model_config"]
    coco_names = config["coco_names"]
    frame_rate = config["frame_rate"]
    input_size = tuple(config["input_size"])
    output_folder = config["output_folder"]
    confidence_threshold = config["confidence_threshold"]
    person_class_id = config["person_class_id"]
    camera_id = config["camera_id"]

    # Load MobileNet-SSD model (Caffe format)
    net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

    # Load COCO class names
    with open(coco_names, "r") as f:
        classes = f.read().strip().split("\n")

    # Initialize video capture
    video_capture = cv2.VideoCapture(camera_id)

    if not video_capture.isOpened():
        print("Error: Could not open video feed.")
        exit()

    # Set desired frame rate
    frame_delay = 1 / frame_rate  # Delay between frames in seconds

    # Variable to track if a person was detected in the previous frame
    person_detected_prev = False

    # Counter for the number of screenshots taken
    screenshot_count = 0

    # Variables to track detection duration
    detection_start_time = None
    detection_duration = 0

    # Variable to track if a 5-second screenshot has been taken
    five_second_screenshot_taken = False

    # Variable to store the filename for the current detection
    current_filename = None

    while True:
        # Get the current date
        today = datetime.now().strftime("%Y-%m-%d")

        # Create a folder for the day if it doesn't exist
        daily_output_folder = os.path.join(output_folder, today)
        os.makedirs(daily_output_folder, exist_ok=True)

        # CSV file to store duration data
        csv_file = os.path.join(daily_output_folder, "detection_durations.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Filename", "Duration (seconds)"])

        # Start timer for frame rate control
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Get frame dimensions
        (h, w) = frame.shape[:2]

        # Prepare the input blob for the model with reduced input size
        blob = cv2.dnn.blobFromImage(frame, 0.007843, input_size, 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Flag to check if a person is detected in the current frame
        person_detected = False

        # Create a copy of the frame for saving (without bounding boxes)
        frame_to_save = frame.copy()

        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter for "person" class
            if confidence > confidence_threshold and int(detections[0, 0, i, 1]) == person_class_id:
                # Set the flag to True if a person is detected
                person_detected = True

                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label on the display frame (not the saved frame)
                label = f"Person: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # If a person was detected in the previous frame but not in the current frame
        if not person_detected and person_detected_prev:
            # Calculate the duration of the detection
            if detection_start_time is not None:
                detection_duration = time.time() - detection_start_time

                # Save the duration to the CSV file
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([current_filename, round(detection_duration, 2)])

                print(f"Person left the frame. Duration: {round(detection_duration, 2)} seconds")

        # If a person is detected, calculate the duration and display it in the live feed
        if person_detected:
            if detection_start_time is None:
                detection_start_time = time.time()  # Initialize start time if not already set
            current_duration = time.time() - detection_start_time
            duration_text = f"Duration: {round(current_duration, 2)}s"
            cv2.putText(frame, duration_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Smaller font size
        else:
            detection_start_time = None  # Reset start time if no person is detected

        # Display the frame with bounding boxes and duration text if show_feed is True
        if show_feed:
            cv2.imshow("Human Detection", frame)

        # Save the frame as soon as a person is detected (without bounding boxes)
        if person_detected and not person_detected_prev:
            # Increment the screenshot counter
            screenshot_count += 1

            # Record the start time of the detection
            detection_start_time = time.time()
            five_second_screenshot_taken = False  # Reset the 5-second screenshot flag

            # Generate a filename with the current timestamp and screenshot count
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            current_filename = os.path.join(daily_output_folder, f"{timestamp}_#{screenshot_count}.jpg")

            # Save the frame without bounding boxes
            cv2.imwrite(current_filename, frame_to_save)
            print(f"Saved: {current_filename} (Total screenshots: {screenshot_count})")

        # If a person is still in the frame
        if person_detected:
            # Calculate the current duration
            current_duration = time.time() - detection_start_time

            # Take another screenshot if the person stays for more than 5 seconds
            if current_duration > 5 and not five_second_screenshot_taken:
                # Increment the screenshot counter
                screenshot_count += 1

                # Generate a filename with the current timestamp and screenshot count
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                current_filename = os.path.join(daily_output_folder, f"{timestamp}_#{screenshot_count}.jpg")

                # Save the frame without bounding boxes
                cv2.imwrite(current_filename, frame_to_save)
                print(f"Saved: {current_filename} (Total screenshots: {screenshot_count})")

                # Mark that the 5-second screenshot has been taken
                five_second_screenshot_taken = True

        # Update the previous detection state
        person_detected_prev = person_detected

        # Calculate elapsed time and enforce frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)

        # Break the loop if 'q' is pressed and show_feed is True
        if show_feed and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up argument parser with a description
    parser = argparse.ArgumentParser(
        description="Webcam Surveillance: Detects humans in a video feed and saves screenshots.",
        epilog="Example usage: python main.py --show --config config.json"
    )

    # Add arguments
    parser.add_argument(
        '--show',
        action='store_true',
        help="Show the video feed during detection (default: False)"
    )
    parser.add_argument(
        '--config',
        default="config.json",
        help="Path to the configuration file (default: config.json)"
    )
    parser.add_argument(
        '--stat',
        action='store_true',
        help="Print the detection statistics from the CSV file"
    )

    # Parse arguments
    args = parser.parse_args()

    # Load configuration from the JSON file
    config = load_config(args.config)

    # Handle --stat argument
    if args.stat:
        today = datetime.now().strftime("%Y-%m-%d")
        output_folder = os.path.join(config["output_folder"], today)
        csv_file = os.path.join(output_folder, "detection_durations.csv")
        print_statistics(csv_file)
        exit()

    # Call the main function with the configuration and show_feed argument
    main(config, show_feed=args.show)
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import time
import csv
import argparse
import json
from tabulate import tabulate  
from collections import deque  

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def print_statistics(output_folder, days=1, top=None):
    """
    Print the CSV file in an organized table for the last 'days' days.
    Starts a new table for each day and shows the header only in the first table.
    Sorts and limits the number of rows displayed for each day if 'top' is specified.
    """
    # Calculate the date range
    today = datetime.now()
    date_range = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    # Collect data from all relevant CSV files
    headers = None
    for date in date_range:
        csv_file = os.path.join(output_folder, date, "detection_durations.csv")
        if os.path.exists(csv_file):
            with open(csv_file, mode="r") as file:
                reader = csv.reader(file)
                if headers is None:
                    headers = next(reader)  # Read the header row (only once)
                else:
                    next(reader)  # Skip the header row for subsequent days
                data = list(reader)  # Read the rest of the data

                # Sort and limit the data only if 'top' is specified
                if top is not None:
                    # Sort data by duration (assuming duration is the second column)
                    data.sort(key=lambda x: float(x[1]), reverse=True)
                    # Limit the number of rows
                    data = data[:top]

                # Print the data in a table format
                print(f"Data for {date}:")
                print(tabulate(data, headers=headers if headers else [], tablefmt="pretty"))
                print()  # Add a blank line between tables
        else:
            print(f"No data available for {date}.")

    if headers is None:
        print("No statistics available for the specified days.")

def delete_directories(output_folder):
    """
    List all date directories in the output folder and allow the user to delete specific directories.
    """
    # Get all date directories in the output folder
    date_dirs = [d for d in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, d))]
    
    if not date_dirs:
        print("No date directories found in the output folder.")
        return

    # Print the list of date directories with indices
    print("Date directories:")
    for idx, dir_name in enumerate(date_dirs, start=1):
        print(f"{idx}. {dir_name}")

    # Prompt the user for input
    user_input = input(
        "Enter the number of the directory to delete (e.g., 1), or a range (e.g., 2-5), or 'q' to quit: "
    ).strip()

    if user_input.lower() == 'q':
        print("Exiting delete mode.")
        return

    try:
        # Handle single directory deletion
        if '-' not in user_input:
            idx = int(user_input)
            if 1 <= idx <= len(date_dirs):
                dir_to_delete = os.path.join(output_folder, date_dirs[idx - 1])
                confirm = input(f"Are you sure you want to delete '{dir_to_delete}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    # Delete the directory and its contents
                    for root, dirs, files in os.walk(dir_to_delete, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(dir_to_delete)
                    print(f"Deleted directory: {dir_to_delete}")
                else:
                    print("Deletion canceled.")
            else:
                print("Invalid input. Please enter a valid number.")

        # Handle range of directories deletion
        else:
            start_idx, end_idx = map(int, user_input.split('-'))
            if 1 <= start_idx <= end_idx <= len(date_dirs):
                confirm = input(
                    f"Are you sure you want to delete directories {start_idx}-{end_idx}? (y/n): "
                ).strip().lower()
                if confirm == 'y':
                    for idx in range(start_idx, end_idx + 1):
                        dir_to_delete = os.path.join(output_folder, date_dirs[idx - 1])
                        # Delete the directory and its contents
                        for root, dirs, files in os.walk(dir_to_delete, topdown=False):
                            for name in files:
                                os.remove(os.path.join(root, name))
                            for name in dirs:
                                os.rmdir(os.path.join(root, name))
                        os.rmdir(dir_to_delete)
                        print(f"Deleted directory: {dir_to_delete}")
                else:
                    print("Deletion canceled.")
            else:
                print("Invalid range. Please enter a valid range.")

    except ValueError:
        print("Invalid input. Please enter a number or a range (e.g., 2-5).")

def main(config, show_feed, no_video):
    try:
        # Load configuration parameters
        model_weights = config["model_weights"]
        model_config = config["model_config"]
        frame_rate = config["frame_rate"]
        input_size = tuple(config["input_size"])
        output_folder = config["output_folder"]
        confidence_threshold = config["confidence_threshold"]
        person_class_id = config["person_class_id"]
        camera_id = config["camera_id"]
        second_screenshot_interval = config["second_screenshot_interval"]
        recording_duration = config["recording_duration"]

        # Constants for camera coverage detection
        COVERAGE_THRESHOLD = config["coverage_threshold"]  
        CONSECUTIVE_FRAMES_THRESHOLD = 30  # Number of consecutive dark frames to trigger an alert
        coverage_frame_buffer = deque(maxlen=CONSECUTIVE_FRAMES_THRESHOLD)

        # Load MobileNet-SSD model (Caffe format)
        net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

        # Initialize video capture
        video_capture = cv2.VideoCapture(camera_id)
        
        if not video_capture.isOpened():
            print("Error: Could not open video feed. Attempting to reconnect...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            video_capture = cv2.VideoCapture(camera_id)
            if not video_capture.isOpened():
                print("Failed to reconnect. Exiting.")
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

        # Variable to track if a second screenshot has been taken
        second_screenshot_taken = False

        # Variable to store the filename for the current detection
        current_filename = None

        # Video recording variables
        video_writer = None
        recording_start_time = None

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
                print("Error: Could not read frame. Attempting to reconnect...")
                video_capture.release()
                time.sleep(5)  # Wait for 5 seconds before retrying
                video_capture = cv2.VideoCapture(camera_id)
                if not video_capture.isOpened():
                    print("Failed to reconnect.")
                    raise RuntimeError("Failed to read frame.")
                continue

            # Check for camera coverage
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            average_brightness = np.mean(gray_frame)

            if average_brightness < COVERAGE_THRESHOLD:
                coverage_frame_buffer.append(1)  # Add 1 to the buffer if the frame is dark
            else:
                coverage_frame_buffer.append(0)  # Add 0 if the frame is not dark

            # Check if the camera is covered (too many consecutive dark frames)
            if sum(coverage_frame_buffer) == CONSECUTIVE_FRAMES_THRESHOLD:
                timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                print(f'Your camera may be covered at {timestamp}.')
                coverage_frame_buffer.clear()  # Reset the buffer after triggering the alert

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
                cv2.imshow("Webcam Surveillance", frame)

            # Save the frame as soon as a person is detected (without bounding boxes)
            if person_detected and not person_detected_prev:
                # Increment the screenshot counter
                screenshot_count += 1

                # Record the start time of the detection
                detection_start_time = time.time()
                second_screenshot_taken = False  # Reset the second screenshot flag

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

                # Take another screenshot if the person stays for more than the specified interval
                if current_duration > second_screenshot_interval and not second_screenshot_taken:
                    # Increment the screenshot counter
                    screenshot_count += 1

                    # Generate a filename with the current timestamp and screenshot count
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    current_filename = os.path.join(daily_output_folder, f"{timestamp}_#{screenshot_count}.jpg")

                    # Save the frame without bounding boxes
                    cv2.imwrite(current_filename, frame_to_save)
                    print(f"Saved: {current_filename} (Total screenshots: {screenshot_count})")

                    # Mark that the second screenshot has been taken
                    second_screenshot_taken = True

                    # Start video recording (if not disabled)
                    if not no_video and video_writer is None:
                        video_filename = os.path.join(daily_output_folder, f"{timestamp}_recording.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (w, h))
                        recording_start_time = time.time()
                        print(f"Started video recording: {video_filename}")

            # Stop video recording if both conditions are met:
            # 1. The recording duration has elapsed.
            # 2. The person has left the frame.
            if video_writer is not None:
                elapsed_recording_time = time.time() - recording_start_time
                if elapsed_recording_time >= recording_duration and not person_detected:
                    video_writer.release()
                    video_writer = None
                    print("Stopped video recording (duration elapsed and person left).")

            # Write the frame to the video file if recording is active
            if video_writer is not None:
                video_writer.write(frame_to_save)

            # Update the previous detection state
            person_detected_prev = person_detected

            # Calculate elapsed time and enforce frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_delay:
                time.sleep(frame_delay - elapsed_time)

            # Break the loop if 'q' is pressed and show_feed is True
            if show_feed and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {str(e)}. Restarting the application...")
        if video_writer is not None:
            video_writer.release()
        video_capture.release()
        cv2.destroyAllWindows()
        time.sleep(5)  # Wait for 5 seconds before restarting
        main(config, show_feed, no_video)
    
    except KeyboardInterrupt:
        print("Exiting...")
        if video_writer is not None:
            video_writer.release()
        video_capture.release()
        cv2.destroyAllWindows()
        
    finally:
        # Release the video capture object and close all OpenCV windows
        if video_writer is not None:
            video_writer.release()
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
        nargs='?',  # Makes --stat optional and allows an optional argument
        const=1,    # Default value if --stat is used without an argument
        type=int,
        help="Print the detection statistics from the CSV file for the last N days (default: 1)"
    )
    parser.add_argument(
        '-top',
        type=int,
        help="Limit the number of rows displayed for each day's statistics"
    )
    parser.add_argument(
        '--novideo',
        action='store_true',
        help="Disable video recording (default: False)"
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help="Enter delete mode to remove specific date directories"
    )

    # Parse arguments
    args = parser.parse_args()

    # Load configuration from the JSON file
    config = load_config(args.config)

    # Handle --delete argument
    if args.delete:
        delete_directories(config["output_folder"])
        exit()

    # Handle --stat argument
    if args.stat is not None:
        print_statistics(config["output_folder"], days=args.stat, top=args.top)
        exit()

    # Call the main function with the configuration and show_feed argument
    main(config, show_feed=args.show, no_video=args.novideo)
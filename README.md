# Webcam Surveillance

This project detects humans in a live webcam feed using a MobileNet-SSD model. It saves screenshots when a person is detected and logs the duration of their presence, also starts video recording if the person stays for too long. The script can be run as a command-line tool on Linux or as a Python script on other platforms.

---

## Features

- **Real-time human detection** using a pre-trained MobileNet-SSD model.
- **Saves screenshots** when a person is detected.
- **Logs detection durations** in a CSV file.
- **Customizable configuration** via `config.json`.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/FarhanSadaf/webcam.git
cd webcam
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### On Linux (Using `webcam` Command)

#### Make the Script Executable:
```bash
chmod +x webcam
```

#### Run the Script:

- **Without showing the video feed:**
  ```bash
  ./webcam
  ```

- **Video Recording Disabled:**
  ```bash
  ./webcam --novideo
  ```

- **With the video feed:**
  ```bash
  ./webcam --show
  ```

- **Print detection statistics:**
  ```bash
  ./webcam --stat
  ```
- **Print detection statistics for the last N days (e.g., last 2 days):**
  ```bash
  ./webcam --stat 2
  ```
- **Print top M results for the last N days (e.g., top 2 results for the last 3 days):**
  ```bash
  ./webcam --stat 2 -top 2
  ```

#### Add to PATH (Optional):
To use `webcam` as a global command:
```bash
sudo mv webcam /usr/local/bin/
```

### On Non-Linux Systems

Run the Python Script Directly:

- **Without showing the video feed:**
  ```bash
  python main.py
  ```

  - **Video Recording Disabled:**
  ```bash
  python main.py --novideo
  ```

- **With the video feed:**
  ```bash
  python main.py --show
  ```

- **Print detection statistics:**
  ```bash
  python main.py --stat
  ```

- **Print detection statistics for the last N days (e.g., last 2 days):**
  ```bash
  python main.py --stat 2
  ```

- **Print top M results for the last N days (e.g., top 2 results for the last 3 days):**
  ```bash
  python main.py --stat 2 -top 2
  ```

---

## Configuration

Modify `config.json` to customize the behavior:

- `model_weights`: Path to the model weights file.
- `model_config`: Path to the model configuration file.
- `frame_rate`: Desired frame rate.
- `input_size`: Input size for the model.
- `output_folder`: Folder to save screenshots and logs.
- `camera_id`: Change video capture camera.
- `second_screenshot_interval`: Time (sec.) after second screenshot will be taken or video recording will be started.
- `recording_duration`: Minimum amount of time (sec.) the video will be recorded.

---

## Example Output

### Screenshots
Screenshots are saved in the `output/<date>/` folder with timestamps.

### CSV Log
Detection durations are logged in `output/<date>/detection_durations.csv`.

### Statistics
Use `--stat` to print the CSV log in a table:
```
+---------------------------+-------------------+
|          Filename         | Duration (seconds)|
+---------------------------+-------------------+
| 2023-10-01_12-34-56_#1.jpg|       3.45        |
| 2023-10-01_12-35-10_#2.jpg|       7.89        |
+---------------------------+-------------------+
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

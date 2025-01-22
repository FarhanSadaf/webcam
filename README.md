
# Human Detection with Webcam

This project detects humans in a live webcam feed using a MobileNet-SSD model. It saves screenshots when a person is detected and logs the duration of their presence. The script can be run as a command-line tool on Linux or as a Python script on other platforms.

---

## Features

- **Real-time human detection** using a pre-trained MobileNet-SSD model.
- **Saves screenshots** when a person is detected.
- **Logs detection durations** in a CSV file.
- **Command-line interface** for easy use on Linux.
- **Customizable configuration** via `config.json`.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/FarhanSadaf/webcam.git
cd your-repo
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
  webcam
  ```

- **With the video feed:**
  ```bash
  webcam --show
  ```

- **Print detection statistics:**
  ```bash
  webcam --stat
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

- **With the video feed:**
  ```bash
  python main.py --show
  ```

- **Print detection statistics:**
  ```bash
  python main.py --stat
  ```

---

## Configuration

Modify `config.json` to customize the behavior:

- `model_weights`: Path to the model weights file.
- `model_config`: Path to the model configuration file.
- `frame_rate`: Desired frame rate.
- `input_size`: Input size for the model.
- `output_folder`: Folder to save screenshots and logs.

---

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Tabulate (`tabulate`) for pretty-printing statistics.

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

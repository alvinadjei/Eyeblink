import sys
import time
import serial
import threading
import random
import sounddevice as sd
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget

# Initialize global constants
num_trials = 10  # number of trials to run
ISI = 0.25  # 250 ms inter-stimulus interval
ITI = 10  # 10 second (on average) inter-trial interval
arduino_port = 'COM4'  # '/dev/cu.usbserial-01C60315'  # Match this to Arduino's port, check by running 'ls /dev/cu.*' in terminal on Mac
baud_rate = 9600  # arduino baud rate
frequency = 880.0  # Frequency in Hz (A5) of CS
tone_duration = 0.28     # Duration in seconds of CS
sample_rate = 44100  # Sample rate in Hz of CS
binary_threshold = 150  # Any pixel value in the processed image below 150 will be set to 0, and above 150 will be set to 1
stability_threshold = 0.25  # FEC value that eye must stay below for at least 200 ms before starting next trial
stability_duration = 0.2  # 200 ms in seconds of stability check

# # Open the file containing the camera's calibration vals
# fs = cv2.FileStorage('Code/IR Camera/calibration/calib_params.xml', cv2.FILE_STORAGE_READ)

# # Read camera calibration vals and save as vars
# mtx = fs.getNode("mtx").mat()
# dist = fs.getNode("dist").mat()
# rvecs = fs.getNode("rvecs").mat()
# tvecs = fs.getNode("tvecs").mat()
# fs.release()

# Establish serial connection
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for the connection to establish

# Mouse ID
mouse_id = input("Please input the mouse's ID: ")

print('Successfully established serial connection to arduino.')

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class ExperimentThread(QThread):
    trial_started = pyqtSignal(int)  # Signal emitted when a trial starts
    trial_completed = pyqtSignal(int)  # Signal emitted when a trial completes
    experiment_finished = pyqtSignal()  # Signal emitted when the experiment ends
    stability_error = pyqtSignal(str)  # Signal for stability check failure
    stim_collector = pyqtSignal(pd.DataFrame)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.num_trials = num_trials
        self.trial_num = 0
        self.running = False
        self.stim_data = pd.DataFrame(columns=["Trial #", "CS Timestamp"])

    def run(self):
        """Main experiment loop"""
        self.running = True
        try:
            for i in range(self.num_trials):
                if not self.running:
                    break
                self.trial_num = i + 1

                # Run a single trial
                self.run_trial(i)

                self.trial_completed.emit(self.trial_num)

        except Exception as e:
            self.stability_error.emit(str(e))

        finally:
            self.running = False
            self.stim_collector.emit(self.stim_data)
            time.sleep(0.5) # Leave time for stim_data to be saved
            self.experiment_finished.emit()

    def run_trial(self, i):
        """Run one trial"""
        # Ensure stability, trigger CS and US, handle timing
        window.ensure_stability(i)  # Make sure to use `MainWindow` methods safely
        self.trial_started.emit(self.trial_num)  # Signal to `MainWindow` that the trial has started
        time.sleep(0.05)  # Simulate pre-CS timing
        cs_timestamp = window.stimuli()  # Execute CS and US
        if self.running:
            time.sleep(ITI + random.uniform(-3,3))  # Inter-trial interval (varies slightly on each trial)

        if cs_timestamp:
            # Save stimulus timestamps to dataframe
            self.stim_data = pd.concat([
                self.stim_data,
                pd.DataFrame([[i+1, cs_timestamp]], columns=self.stim_data.columns)
            ], ignore_index=True)

    def stop(self):
        """Gracefully stop the thread"""
        self.running = False
        print("Completing current trial, then stopping...")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEC Measurement")

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

         # Horizontal layout for video feed and explanations
        video_and_text_layout = QHBoxLayout()

        # Video display
        self.video_label = QLabel()
        self.video_label.setMouseTracking(True)  # Enable mouse tracking
        self.video_label.mousePressEvent = self.mousePressEvent  # Set mouse press func
        self.video_label.mouseMoveEvent = self.mouseMoveEvent  # Set mouse movement func
        self.video_label.mouseReleaseEvent = self.mouseReleaseEvent  # Set mouse release func
        self.video_label.mouseReleaseEvent = self.mouseReleaseEvent  # Set mouse release func
        video_and_text_layout.addWidget(self.video_label)

        # Explanation text
        self.explanation_label = QLabel("""
            This experiment involves combining a conditioned stimulus (a short tone) and an unconditioned stimulus (puff of air in the eye) to a mouse. The buttons below control various aspects of the experiment:

            - Start Experiment Button: Starts the experiment with 
              predefined settings.
            - Stop Experiment Button: The experiment will finish 
              current trial, then stop.

            Mouse controls:
            - Left mouse button: select eye area (FEC measured 
              within this ellipse)
            - Right mouse button: select area to zoom into

            Key controls (set lighting BEFORE starting 
            experiment):
            - R: reset FEC roi and zoom out
            - H: Toggle house lights on/off
            - A: Decrease side IR LED's brightness
            - D: Increase side IR LED's brightness
            - S: Decrease top IR LED's brightness
            - W: Increase top IR LED's brightness
            - J: Decrease head-on IR LED's brightness
            - K: Increase head-on IR LED's brightness
            """
        )
        self.explanation_label.setWordWrap(True)  # Wrap the text to fit the width
        self.explanation_label.setFixedWidth(300)  # Set a fixed width to control text box size
        video_and_text_layout.addWidget(self.explanation_label)

        # Add the horizontal layout to the main layout
        self.layout.addLayout(video_and_text_layout)

        # Start/Stop button
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        self.layout.addWidget(self.start_button)

        # Integrate CameraThread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)

        # Current frame
        self.current_frame = None

        # Data
        self.fec_data = pd.DataFrame(columns=["Timestamp", "Trial #", "FEC"])
        self.stim_data = pd.DataFrame(columns=["Trial #", "CS Timestamp"])
        self.trial_num = 0
        
        # Ellipse data
        self.drawing_ellipse = False
        self.ellipse_start = None
        self.ellipse_end = None
        self.ellipse_params = None

        # Save current FEC value
        self.fec_value = 0
        
        # Rectangle data
        self.drawing_rect = False
        self.rect_start = None
        self.rect_end = None
        self.rect_params = None
        
        # Zoom values
        self.top_left_zoom = None
        self.bottom_right_zoom = None

        # Flags specifying if trial/experiment is running
        self.trial_in_progress = False
        self.experiment_running = False

        # Integrate ExperimentThread
        self.experiment_thread = ExperimentThread()
        self.experiment_thread.trial_started.connect(self.on_trial_started)
        self.experiment_thread.trial_completed.connect(self.on_trial_completed)
        self.experiment_thread.experiment_finished.connect(self.on_experiment_finished)
        self.experiment_thread.stability_error.connect(self.on_stability_error)
        self.experiment_thread.stim_collector.connect(self.save_stim_data)
    
    def keyPressEvent(self, event):
        # Lighting cannot be changed once experiment begins
        if not self.experiment_running:
            """Handle key press events to control lighting."""
            if event.key() == Qt.Key_H:  # Check if the 'H' key is pressed
                try:
                    ser.write(b'h')  # Send 'h' to the Arduino, toggle houselights on/off
                    response = int(ser.readline().decode().strip())  # Read confirmation
                    print(f"Houselight turned {response}")
                except Exception as e:
                    print(f"Error sending 'h' to Arduino: {e}")
            
            elif event.key() == Qt.Key_A:  # Check if the 'A' key is pressed
                try:
                    ser.write(b'a')  # Send 'a' to the Arduino, turn side IR led brightness down
                    response = int(ser.readline().decode().strip())  # Read confirmation
                    if response != -1:
                        print(f"Side IR LED brightness: {int(response / 255 * 100)}%")
                except Exception as e:
                    print(f"Error sending 'a' to Arduino: {e}")
            
            elif event.key() == Qt.Key_D:  # Check if the the 'D' key is pressed
                try:
                    ser.write(b'd')  # Send 'd' to the Arduino, turn side IR led brightness up
                    response = int(ser.readline().decode().strip())  # Read confirmation
                    if response != -1:
                        print(f"Side IR LED brightness: {int(response / 255 * 100)}%")
                except Exception as e:
                    print(f"Error sending 'd' to Arduino: {e}")

            elif event.key() == Qt.Key_S:  # Check if the 'S' key is pressed
                try:
                    ser.write(b's')  # Send 's' to the Arduino, turn top IR led brightness down
                    response = int(ser.readline().decode().strip())  # Read confirmation
                    if response != -1:
                        print(f"Top IR LED brightness: {int(response / 255 * 100)}%")
                except Exception as e:
                    print(f"Error sending 's' to Arduino: {e}")
            
            elif event.key() == Qt.Key_W:  # Check if the 'W' key is pressed
                try:
                    ser.write(b'w')  # Send 'w' to the Arduino, turn top IR led brightness up
                    response = int(ser.readline().decode().strip())  # Read confirmation
                    if response != -1:
                        print(f"Top IR LED brightness: {int(response / 255 * 100)}%")
                except Exception as e:
                    print(f"Error sending 'w' to Arduino: {e}")
            
            elif event.key() == Qt.Key_J:  # Check if the 'J' key is pressed
                try:
                    ser.write(b'j')  # Send 'j' to the Arduino, turn head-on IR led brightness down
                    response = int(ser.readline().decode().strip())  # Read confirmation
                    if response != -1:
                        print(f"Head-on IR LED brightness: {int(response / 255 * 100)}%")
                except Exception as e:
                    print(f"Error sending 'j' to Arduino: {e}")

            elif event.key() == Qt.Key_K:  # Check if the 'K' key is pressed
                try:
                    ser.write(b'k')  # Send 'k' to the Arduino, turn head-on IR led brightness up
                    response = int(ser.readline().decode().strip())  # Read confirmation
                    if response != -1:
                        print(f"Head-on IR LED brightness: {int(response / 255 * 100)}%")
                except Exception as e:
                    print(f"Error sending 'k' to Arduino: {e}")

            elif event.key() == Qt.Key_R: # Check if the 'R' key is pressed, reset zoom and roi selection
                # Reset ellipse/roi
                self.ellipse_start = None
                self.ellipse_end = None
                self.ellipse_params = None
                # Reset rectangle/zoom
                self.rect_start = None
                self.rect_end = None
                self.rect_params = None
                self.top_left_zoom = None
                self.bottom_right_zoom = None

        else:
            # Call the base class implementation
            super().keyPressEvent(event)

    def scale_coords(self, event):
        widget_width, widget_height = self.video_label.width(), self.video_label.height()
        image_height, image_width = self.current_frame.shape[:2]

        x_scale = image_width / widget_width
        y_scale = image_height / widget_height

        x = int(event.pos().x() * x_scale)
        y = int(event.pos().y() * y_scale)

        return x, y

    def mousePressEvent(self, event):
        """Handle left click onset to begin drawing ellipse if experiment is not running
        """
        if not self.experiment_running:
            if event.button() == Qt.LeftButton:  # Left click --> Start drawing ellipse
                # Scale the mouse coordinates to image coordinates
                self.ellipse_start = self.scale_coords(event)
                self.drawing_ellipse = True  # Begin drawing
            elif event.button() == Qt.RightButton:  # Right click --> Start drawing rectangle
                self.rect_start = self.scale_coords(event)
                self.drawing_rect = True

    
    def mouseMoveEvent(self, event):
        """Handle mouse movement to draw ellipse if experiment is not running
        """
        if not self.experiment_running:
            if self.drawing_ellipse:
                self.ellipse_end = self.scale_coords(event)
                # Calculate ellipse parameters
                self.calculate_ellipse(release=False)
                
            if self.drawing_rect:
                self.rect_end = self.scale_coords(event)
                # Calculate rectangle parameters
                self.calculate_rectangle(release=False)

    def mouseReleaseEvent(self, event):
        """Handle left click release to stop drawing ellipse if experiment is not running
        """
        if not self.experiment_running:
            if event.button() == Qt.LeftButton:
                self.ellipse_end = self.scale_coords(event)
                self.drawing_ellipse = False  # End drawing

                # Calculate ellipse parameters
                self.calculate_ellipse(release=True)
            elif event.button() == Qt.RightButton:
                self.rect_end = self.scale_coords(event)
                self.drawing_rect = False
                self.calculate_rectangle(release=True)
    
    def calculate_ellipse(self, release):
        if self.ellipse_start and self.ellipse_end:
            x1, y1 = self.ellipse_start
            x2, y2 = self.ellipse_end

            # Center of the ellipse
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # Axes lengths
            axes = (abs(x2 - x1), abs(y2 - y1))
            # Angle (assume 0 for simplicity)
            angle = 0

            self.ellipse_params = (center, axes, angle)

            if release:
                if x1 == x2 or y1 == y2:
                    self.ellipse_start = None
                    self.ellipse_end = None
                    self.ellipse_params = None
                    return
            print(f"Ellipse drawn with center={center}, axes={axes}, angle={angle}")
    
    def calculate_rectangle(self, release):
        """Calculate the rectangle parameters."""
        if self.rect_start and self.rect_end:
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
        
            top_left = (min(x1, x2), min(y1, y2))
            bottom_right = (max(x1, x2), max(y1, y2))

            self.rect_params = (top_left, bottom_right)

            if release:
                # Calculate rectangle width and height
                rect_width = bottom_right[0] - top_left[0]
                rect_height = bottom_right[1] - top_left[1]

                # Get the aspect ratio of the original frame
                frame_height, frame_width = self.current_frame.shape[:2]
                aspect_ratio = frame_width / frame_height
                
                # Avoid division by zero
                if rect_width == 0 or rect_height == 0:
                    # print("Invalid rectangle: zero width or height.")
                    self.rect_start = None
                    self.rect_end = None
                    self.rect_params = None
                    return

                # Adjust rectangle dimensions to maintain aspect ratio
                if rect_width / rect_height > aspect_ratio:
                    # Adjust height to match aspect ratio
                    new_height = int(rect_width / aspect_ratio)
                    height_diff = new_height - rect_height
                    top_left = (top_left[0], max(0, top_left[1] - height_diff // 2))
                    bottom_right = (bottom_right[0], min(frame_height, bottom_right[1] + height_diff // 2))
                else:
                    # Adjust width to match aspect ratio
                    new_width = int(rect_height * aspect_ratio)
                    width_diff = new_width - rect_width
                    top_left = (max(0, top_left[0] - width_diff // 2), top_left[1])
                    bottom_right = (min(frame_width, bottom_right[0] + width_diff // 2), bottom_right[1])

                # Map rectangle coordinates from the zoomed frame to the original frame
                if self.top_left_zoom and self.bottom_right_zoom:
                    zoom_x1, zoom_y1 = self.top_left_zoom
                    zoom_x2, zoom_y2 = self.bottom_right_zoom

                    # Calculate aspect ratio of original frame
                    frame_width = self.current_frame.shape[1]
                    frame_height = self.current_frame.shape[0]
                    
                    # Scale factors for the zoomed frame
                    zoom_width = zoom_x2 - zoom_x1
                    zoom_height = zoom_y2 - zoom_y1

                    scale_x = zoom_width / frame_width
                    scale_y = zoom_height / frame_height

                    # Map rectangle coordinates back to the original frame
                    top_left_original = (
                        int(zoom_x1 + top_left[0] * scale_x),
                        int(zoom_y1 + top_left[1] * scale_y),
                    )
                    bottom_right_original = (
                        int(zoom_x1 + bottom_right[0] * scale_x),
                        int(zoom_y1 + bottom_right[1] * scale_y),
                    )
                else:
                    # No zoom applied; use original frame coordinates
                    top_left_original = top_left
                    bottom_right_original = bottom_right
                    
                print(f"Rectangle drawn with top_left={top_left_original}, bottom_right={bottom_right_original}")
                
                # Save zoom values
                self.top_left_zoom = top_left_original
                self.bottom_right_zoom = bottom_right_original

                # Reset rectangle parameters
                self.rect_start = None
                self.rect_end = None
                self.rect_params = None
    
    def update_frame(self, frame):
        if self.top_left_zoom and self.bottom_right_zoom:  # If rectangle has been drawn
            # Crop the region of interest
            x_min, y_min = self.top_left_zoom
            x_max, y_max = self.bottom_right_zoom
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # Resize the cropped frame to match the original display size
            zoomed_frame = cv2.resize(cropped_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Update the displayed frame
            self.current_frame = zoomed_frame
        else:
            self.current_frame = frame
            
        if self.ellipse_params:
            # Draw the ellipse on the frame
            cv2.ellipse(self.current_frame, self.ellipse_params, (0, 255, 0), 2)
            # Perform FEC calculation here
            fec_value = self.calculate_fec(self.current_frame)
            # Display FEC value in the top left corner
            cv2.putText(self.current_frame, f"FEC: {fec_value:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 3)
            
        # Draw the rectangle
        if self.rect_params:
            top_left, bottom_right = self.rect_params
            cv2.rectangle(self.current_frame, top_left, bottom_right, (255, 0, 0), 2)
            
        if self.experiment_running and self.trial_in_progress:
            timestamp = pd.Timestamp.now()
            self.fec_data = pd.concat([
                self.fec_data,
                pd.DataFrame([[timestamp, self.trial_num, fec_value]], columns=self.fec_data.columns)
            ], ignore_index=True)
            
        # Convert the frame to QImage for display
        rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qimage = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def calculate_fec(self, frame):
        if self.ellipse_params:
            # FEC calculation logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            filtered_gray = cv2.medianBlur(gray, 5)  # Apply 5 x 5 median filter to eliminate "salt and pepper" noise
            # Apply a binary threshold to the grayscale image
            _, binary = cv2.threshold(filtered_gray, binary_threshold, 255, cv2.THRESH_BINARY)  # Any pixel value < binary_threshold is set to 0, and > binary_threshold is 255
            
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.ellipse(mask, self.ellipse_params, 255, -1)
            roi = cv2.bitwise_and(binary, binary, mask=mask)

            lighter_pixels = cv2.countNonZero(roi)
            total_pixels = cv2.countNonZero(mask)
            self.fec_value = lighter_pixels / total_pixels if total_pixels > 0 else 0
            return self.fec_value
        return 0
    
    def start_experiment(self):
        if not self.experiment_running:
            
            if self.ellipse_params is not None:
                # Start experiment thread
                self.experiment_running = True
                self.start_button.setText("Stop Experiment")
                self.experiment_thread.num_trials = num_trials
                self.experiment_thread.start()
            else:
                print('Please draw roi before beginning experiment...')
            
        else:
            self.stop_experiment()
    
    def stop_experiment(self):
        """Stop the experiment gracefully"""
        self.experiment_running = False
        self.start_button.setText("Start Experiment")
        self.experiment_thread.stop()
    
    def ensure_stability(self, i):
        """Check if FEC stays above 0.75 for at least 200 ms
        """
        start_time = None

        print("Checking condition: FEC stays below 0.25 for at least 200 ms")

        while self.experiment_running:
            if self.fec_value < stability_threshold:  # If eye is >= 75% open (or < 25% closed)
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= stability_duration:  # Break loop if eyes stays open longer than 200 ms
                    print(f"Condition met: commencing Trial {i+1}.")
                    break
            else:
                start_time = None  # Reset if the mouse blinks

            # Check every 10 ms
            time.sleep(0.01)
    
    def stimuli(self):
        if self.experiment_running:
            ser.write(b'T')  # send 'T' command to Arduino to trigger trial
            response = ser.readline().decode().strip()  # Read confirmation
            if response == 'd':  # check for valid confirmation message
                timestamp = pd.Timestamp.now()     # Record when response is received, arduino code has completed execution
                time.sleep(0.05)  # Wait for air puff to complete
                return timestamp
            else:
                raise ValueError(f"Unexpected response from Arduino: {response}")
    
    def on_trial_started(self, trial_num):
        self.trial_num = trial_num
        self.trial_in_progress = True
        # print(f"Trial {trial_num} started...")

    def on_trial_completed(self, trial_num):
        self.trial_in_progress = False
        print(f"Trial {trial_num} completed.")

    def on_experiment_finished(self):
        self.experiment_running = False
        self.start_button.setText("Start Experiment")
        
        print("Experiment finished!")

    def on_stability_error(self, error_message):
        print(f"Stability check failed: {error_message}")        

    def save_stim_data(self, stim_data):
        self.stim_data = stim_data

    def closeEvent(self, event):
        if self.experiment_thread.isRunning():
            self.experiment_thread.stop()
            self.experiment_thread.wait()
        self.camera_thread.stop()
        self.camera_thread.stop()

        # Save data to csv's
        if not self.fec_data.empty:
            self.fec_data.to_csv(f"Code/capture/FEC/mouse_{mouse_id}_fec.csv", index=False)
        if not self.stim_data.empty:
            self.stim_data.to_csv(f"Code/capture/stim/mouse_{mouse_id}_stim.csv", index=False)
        
        super().closeEvent(event)
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.camera_thread.start()
    sys.exit(app.exec_())

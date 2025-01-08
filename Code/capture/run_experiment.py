import sys
import time
import serial
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
ITI = 10  # 10 second inter-trial interval
arduino_port = 'COM4'  # '/dev/cu.usbserial-01C60315'  # Match this to Arduino's port, check by running 'ls /dev/cu.*' in terminal on Mac
baud_rate = 9600  # arduino baud rate
frequency = 880.0  # Frequency in Hz (A5) of CS
tone_duration = 0.25     # Duration in seconds of CS
sample_rate = 44100  # Sample rate in Hz of CS
binary_threshold = 50  # Any pixel value in the processed image below 150 will be set to 0, and above 150 will be set to 1
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
        self.stim_data = pd.DataFrame(columns=["Trial #", "CS Timestamp", "US Timestamp"])

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
        cs_timestamp = window.cond_stim()  # ISI baked into conditioned stimulus func
        us_timestamp = window.uncond_stim()
        if self.running:
            time.sleep(ITI)  # Inter-trial interval

        if cs_timestamp and us_timestamp:
            # Save stimulus timestamps to dataframe
            self.stim_data = pd.concat([
                self.stim_data,
                pd.DataFrame([[i+1, cs_timestamp, us_timestamp]], columns=self.stim_data.columns)
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

            Key controls (set BEFORE starting experiment):
            - H: Toggle house lights on/off
            - A: Decrease side IR LED's brightness
            - D: Increase side IR LED's brightness
            - S: Decrease top IR LED's brightness
            - W: Increase top IR LED's brightness
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
        self.stim_data = pd.DataFrame(columns=["Trial #", "CS Timestamp", "US Timestamp"])
        self.trial_num = 0
        
        # Ellipse data
        self.drawing = False
        self.ellipse_start = None
        self.ellipse_end = None
        self.ellipse_params = None
        self.trial_in_progress = False

        # Save current FEC value
        self.fec_value = 0

        # Flag specifying if experiment is running
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
        if event.button() == Qt.LeftButton and not self.experiment_running:
            # Scale the mouse coordinates to image coordinates
            self.ellipse_start = self.scale_coords(event)
            self.drawing = True  # Begin drawing
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement to draw ellipse if experiment is not running
        """
        if self.drawing and not self.experiment_running:
            self.ellipse_end = self.scale_coords(event)

            # Calculate ellipse parameters
            self.calculate_ellipse(release=False)

    def mouseReleaseEvent(self, event):
        """Handle left click release to stop drawing ellipse if experiment is not running
        """
        if event.button() == Qt.LeftButton and not self.experiment_running:
            self.ellipse_end = self.scale_coords(event)
            self.drawing = False  # End drawing

            # Calculate ellipse parameters
            self.calculate_ellipse(release=True)
    
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
                print(f"Ellipse drawn with center={center}, axes={axes}, angle={angle}")
    
    def update_frame(self, frame):
        self.current_frame = frame
        if self.ellipse_params:
            # Draw the ellipse on the frame
            cv2.ellipse(frame, self.ellipse_params, (0, 255, 0), 2)
            # Perform FEC calculation here
            fec_value = self.calculate_fec(frame)
            # Display FEC value in the top left corner
            cv2.putText(frame, f"FEC: {fec_value:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 3)

        if self.experiment_running and self.trial_in_progress:
            timestamp = pd.Timestamp.now()
            self.fec_data = pd.concat([
                self.fec_data,
                pd.DataFrame([[timestamp, self.trial_num, fec_value]], columns=self.fec_data.columns)
            ], ignore_index=True)
            
        # Convert the frame to QImage for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
    def cond_stim(self):
        """Executes conditioned stimulus (plays the musical tone A5 for 300ms)
        """
        if self.experiment_running:
            # Generate the sound wave
            t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
            waveform = 0.5 * np.sin(2 * np.pi * frequency * t)  # 0.5 for volume control

            # Record the CS timestamp using cv2.getTickCount()
            timestamp = pd.Timestamp.now()

            # Play the sound
            sd.play(waveform, samplerate=sample_rate)
            sd.wait()  # Wait until the sound has finished playing
            
            # Return timestamp of stimulus onset
            return timestamp

    def uncond_stim(self):
        """Executes unconditioned stimulus (triggers airpuff)
        """
        if self.experiment_running:
            ser.write(b'p')  # send 'p' command to Arduino to trigger the puff
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

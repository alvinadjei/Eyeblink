import sys
import time
import serial
import threading
import sounddevice as sd
import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QPushButton, QWidget

# Initialize global constants
num_trials = 10  # number of trials to run
ISI = 0.25  # 250 ms inter-stimulus interval
ITI = 10  # 10 second inter-trial interval
arduino_port = 'COM4'  # '/dev/cu.usbserial-01C60315'  # Match this to Arduino's port, check by running 'ls /dev/cu.*' in terminal on Mac
baud_rate = 9600  # arduino baud rate
frequency = 880.0  # Frequency in Hz (A5) of CS
tone_duration = 0.3     # Duration in seconds of CS
sample_rate = 44100  # Sample rate in Hz of CS
binary_threshold = 150  # Any pixel value in the processed image below 150 will be set to 0, and above 150 will be set to 1
stability_threshold = 0.25  # FEC value that eye must stay below for at least 200 ms before starting next trial
stability_duration = 0.2  # 200 ms in seconds of stability check

# Open the file containing the camera's calibration vals
fs = cv2.FileStorage('Code/IR Camera/calibration/calib_params.xml', cv2.FILE_STORAGE_READ)

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEC Measurement")

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Video display
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        # Start/Stop button
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        self.layout.addWidget(self.start_button)

        # Camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)

        # Data
        self.fec_data = pd.DataFrame(columns=["Timestamp", "Trial #", "FEC"])
        self.stim_data = pd.DataFrame(columns=["Trial #", "CS Timestamp", "US Timestamp"])
        self.trial_num = 0
        
        # Ellipse data
        self.drawing_ellipse = False
        self.ellipse_params = None
        self.experiment_running = False
        self.trial_in_progress = False

    def update_frame(self, frame):
        if self.drawing_ellipse and self.ellipse_params:
            # Draw the ellipse on the frame
            cv2.ellipse(frame, self.ellipse_params, (0, 255, 0), 2)

        # Convert the frame to QImage for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qimage = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

        if self.experiment_running and self.trial_in_progress:
            # Perform FEC calculation here
            fec_value = self.calculate_fec(frame)
            timestamp = pd.Timestamp.now()
            self.fec_data = pd.concat([
                self.fec_data,
                pd.DataFrame([[timestamp, self.trial_num, fec_value]], columns=self.fec_data.columns)
            ], ignore_index=True)

    def calculate_fec(self, frame):
        if self.ellipse_params:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, self.ellipse_params, 255, -1)
            roi = cv2.bitwise_and(frame, frame, mask=mask)

            # FEC calculation logic
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lighter_pixels = np.sum(gray > 128)
            total_pixels = np.sum(mask > 0)
            fec_value = lighter_pixels / total_pixels if total_pixels > 0 else 0
            return fec_value
        return 0
    
    def run_experiment(self):
        # Run experiment
        for i in range(num_trials):
            self.trial_num = i+1
            self.run_trial(i)

    def start_experiment(self):
        if not self.experiment_running:
            self.start_button.setText("Stop Experiment")
            self.experiment_running = True
            
            experiment_thread = threading.Thread(target=self.run_experiment)
            experiment_thread.start()
            
        else:
            self.start_button.setText("Start Experiment")
            self.experiment_running = False

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.fec_data.to_csv(f"FEC/mouse_{mouse_id}_fec.csv", index=False)
        super().closeEvent(event)
    
    def run_trial(self, i):
        """Run one trial

        Args:
            i (int): Trial #
        """
        try:
            # Begin trial
            self.trial_in_progress = True
            
            # Ensure mouse eye has been open for at least 200ms before beginning trial
            self.ensure_stability()
            
            # Start recording 50 ms before stimulus
            time.sleep(0.05)
            
            # Conditioned Stimulus
            cs_timestamp = self.cond_stim()  # returns timestamp of conditioned stimulus onset
            time.sleep(ISI)  # 250ms ISI
            
            # Unconditioned stimulus
            us_timestamp = self.uncond_stim()  # returns timestamp of unconditioned stimulus onset
            
            # Wait for air puff to complete
            time.sleep(0.05)
                        
            # Initiate ITI
            print("Waiting 10 seconds in between trials...")
            time.sleep(ITI)  # 10s ITI
            
            self.stim_data = pd.concat([
                self.stim_data,
                pd.DataFrame([[i+1, cs_timestamp, us_timestamp]], columns=self.stim_data.columns)
            ], ignore_index=True)
            
            # End trial
            self.trial_in_progress = False
            
        except KeyboardInterrupt:
            print("Puff sequence interrupted by user.")
    
    def ensure_stability(self):
        """Check if FEC stays above 0.75 for at least 200 ms
        """
        #TODO: Implement this function
        pass
    
    def cond_stim(self):
        """Executes conditioned stimulus (plays the musical tone A5 for 300ms)
        """
        # Generate the sound wave
        t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)  # 0.5 for volume control

        # Record the CS timestamp using cv2.getTickCount()
        timestamp = pd.Timestamp.now()

        # Play the sound
        sd.play(waveform, samplerate=sample_rate)
        # sd.wait()  # Wait until the sound has finished playing
        
        # Return timestamp of stimulus onset
        return timestamp

    def uncond_stim(self):
        """Executes unconditioned stimulus (triggers airpuff)
        """
        ser.write(b'p')  # send 'p' command to Arduino to trigger the puff
        response = ser.readline().decode().strip()  # Read confirmation
        timestamp = pd.Timestamp.now()     # Record when response is received, arduino code has completed execution
        return timestamp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.camera_thread.start()
    sys.exit(app.exec_())

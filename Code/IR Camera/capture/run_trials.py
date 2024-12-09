import sys
import time
import serial
import threading
from multiprocessing import Process, Queue
import sounddevice as sd
import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

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

# Read camera calibration vals and save as vars
mtx = fs.getNode("mtx").mat()
dist = fs.getNode("dist").mat()
rvecs = fs.getNode("rvecs").mat()
tvecs = fs.getNode("tvecs").mat()
fs.release()

# Establish serial connection
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for the connection to establish

# Mouse ID
mouse_id = input("Please input the mouse's ID: ")

print('Successfully established serial connection to arduino.')

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize camera and timer
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Set to ~30 FPS
        
        # Booleans that tell code when to run
        self.trial_in_progress = False  # Boolean that describes whether a trial is in progress
        self.running = False  # Boolean that describes whether an experiment is in progress

        # Which number trial we are on
        self.trial_ind = 0
        
        # Current processed frame
        self.frame = None
        
        # Set up the GUI
        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)
        self.setWindowTitle("Mouse Eye Cam")
        self.resize(800, 600)

        # Ellipse parameters
        self.ellipse_start = None
        self.ellipse_end = None
        self.drawing = False
        self.light_fraction = None

        # Keep track of CS timestamp
        self.cs_tick_count = None
        
        # Dataframe to hold fec values w/ timestamps
        self.df_fec = pd.DataFrame(columns=['Current Timestamp', 'Trial #', 'FEC',])  # Create df to hold FEC values
        
        # Dataframe to hold CS and US timestamps
        self.df_stim = pd.DataFrame(columns=['Trial #', 'CS Timestamp', 'US Timestamp'])  # Create df to hold CS and US timestamps for each trial

    def start_fec_process(self):
        """Start the FEC acquisition process."""
        self.running_event.set()
        self.fec_process = Process(
            target=fec_acquisition,
            args=(self.fec_queue, self.running_event, self.trial_in_progress_event, self.get_frame),
        )
        self.fec_process.start()

    def stop_fec_process(self):
        """Stop the FEC acquisition process."""
        self.running_event.clear()
        self.fec_process.join()
        self.fec_process = None
    
    def get_frame(self):
        """Retrieve the current processed frame for the FEC process."""
        return self.frame if self.frame is not None else None

    def update_frame(self):
        """Update and show frame, elliptical ROI, and FEC value
        """
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray_frame.shape[:2]
        newcameramtx, dst_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # Undistort fisheye image
        dst = cv2.undistort(gray_frame, mtx, dist, None, newcameramtx)

        # Crop the image
        x, y, w, h = dst_roi
        dst = dst[y:y+h, x:x+w]
        
        # Apply 5 x 5 median filter to eliminate "salt and pepper" noise
        filtered_dst = cv2.medianBlur(dst, 5)
        
        # Apply a binary threshold to the grayscale image
        _, binary_dst = cv2.threshold(filtered_dst, binary_threshold, 255, cv2.THRESH_BINARY)  # Any pixel value < binary_threshold is set to 0, and > binary_threshold is 255
        
        # Save processed image as class attribute to be used in other functions
        self.frame = binary_dst
        
        # Convert back to 3-channel grayscale for display (to avoid single-channel issues with QImage)
        display_frame = cv2.merge([binary_dst] * 3)

        # Dynamically resize the window to match the image size
        window_width, window_height = display_frame.shape[1], display_frame.shape[0]
        self.resize(window_width, window_height)

        # Draw ellipse if defined
        if self.ellipse_start and self.ellipse_end:
            start_x, start_y = self.ellipse_start
            end_x, end_y = self.ellipse_end

            center = ((start_x + end_x) // 2, (start_y + end_y) // 2)
            axes = (abs(end_x - start_x) // 2, abs(end_y - start_y) // 2)
            cv2.ellipse(display_frame, center, axes, 0, 0, 360, (255, 0, 255), 2)  # Draw purple ellipse

            # Create a single-channel mask for the ellipse area
            mask = np.zeros(binary_dst.shape, dtype=np.uint8)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            # Calculate the fraction of lighter pixels inside the ellipse using single-channel image
            masked_roi = cv2.bitwise_and(binary_dst, binary_dst, mask=mask)  # Apply the fixed mask to the current frame
            total_pixels = cv2.countNonZero(mask)  # Total pixels in the ellipse
            light_pixels = cv2.countNonZero(masked_roi)  # Light pixels within the ellipse
            self.light_fraction = (light_pixels / total_pixels) if total_pixels > 0 else 0  # Fraction of light pixels within the ellipse

            # Display FEC value in the top left corner
            cv2.putText(display_frame, f"FEC: {self.light_fraction:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 3)

        # Convert image to QImage
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update the QLabel with the new frame
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def mousePressEvent(self, event):
        """Handle left click onset to begin drawing ellipse if experiment is not running
        """
        if event.button() == Qt.LeftButton and not self.running:
            # Scale the mouse coordinates to image coordinates
            self.ellipse_start = (int(event.position().x()), int(event.position().y()))
            self.drawing = True  # Begin drawing
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement to draw ellipse if experiment is not running
        """
        if self.drawing and not self.running:
            self.ellipse_end = (int(event.position().x()), int(event.position().y()))

    def mouseReleaseEvent(self, event):
        """Handle left click release to stop drawing ellipse if experiment is not running
        """
        if event.button() == Qt.LeftButton and not self.running:
            self.ellipse_end = (int(event.position().x()), int(event.position().y()))
            self.drawing = False  # End drawing
        
    def keyPressEvent(self, event):
        """Press spacebar to begin experiment
        """
        if event.key() == Qt.Key_Space:  # Spacebar pressed
            if self.ellipse_start and self.ellipse_end:  # Check if ellipse is drawn
                if not self.running:  # Only start a new thread if not already running
                    self.running = True
                    print('Beginning experiment...')
                    self.start_fec_process()  # Start the FEC process
                    threading.Thread(target=self.__stimuli).start()
                else:
                    self.running = False  # Stop experiment
                    print('Stopping experiment...')
                    self.stop_fec_process()  # Stop the FEC process
                    self.close()
            else:
                print("Please draw an ellipse around the mouse's eye, then press the spacebar to begin.")
    
    def calculate_fec(self, frame):
        """Calculate the light fraction from the given frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        fec = cv2.countNonZero(binary) / (binary.shape[0] * binary.shape[1])
        return fec

    def fec_acquisition(self, queue, running_event, trial_in_progress_event, frame_getter):
        """Function to run FEC acquisition in a separate process."""
        while running_event.is_set():
            if trial_in_progress_event.is_set():
                frame = frame_getter()
                if frame is not None:
                    fec_tick_count = cv2.getTickCount()
                    timestamp = fec_tick_count / cv2.getTickFrequency() * 1000  # Convert to milliseconds
                    # Assume `calculate_fec()` is a helper function to compute the FEC from the frame.
                    self.light_fraction = self.calculate_fec(frame)  
                    queue.put((timestamp, self.light_fraction))
            time.sleep(0.035)  # Approx 30 FPS
    
    def __save_current_fec(self):
        """Save the current FEC to a pandas df
        """
        while self.running:
            try:
                timestamp, fec_value = self.fec_queue.get_nowait()  # Non-blocking retrieval
                self.df_fec.loc[len(self.df_fec)] = [timestamp, self.trial_ind, fec_value]
            except:
                pass  # Queue is empty, continue
            time.sleep(0.05)  # Small delay to avoid CPU overload

    def __cond_stim(self):
        """Executes conditioned stimulus (plays the musical tone A5 for 300ms)
        """
        # Generate the sound wave
        t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)  # 0.5 for volume control

        # Record the CS timestamp using cv2.getTickCount()
        self.cs_tick_count = cv2.getTickCount()

        # Play the sound
        sd.play(waveform, samplerate=sample_rate)
        # sd.wait()  # Wait until the sound has finished playing
    
    def __ensure_stability(self):
        """Check if FEC stays above 0.75 for at least 200 ms
        """
        start_time = None

        while self.running:
            if self.light_fraction < stability_threshold:  # If eye is >= 75% open (or < 25% closed)
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= stability_duration:  # Break loop if eyes stays open longer than 200 ms
                    print("Condition met: FEC stayed above 0.75 for at least 200 ms, commencing next trial.")
                    break
            else:
                start_time = None  # Reset if the mouse blinks

            # Check every 10 ms
            time.sleep(0.01)
    
    def __stimuli(self):
        """Run eyeblink experiment
        """
        for i in range(num_trials):
            if self.running == False:  # If space bar is pressed during experiment, finish current trial then stop
                print("Experiment ended early by user")
                break
            print(f"Trial {i+1} in progress")  # Print trial number
            self.trial_ind = i  # Keep track of trial number
            try:
                self.__ensure_stability()  # Ensure mouse eye has been open for at least 200ms before beginning trial
                self.trial_in_progress = True  # Trial begins
                time.sleep(0.05)  # Start recording 50 ms before stimulus
                self.__cond_stim()  # Conditioned stimulus
                time.sleep(ISI)  # 250ms ISI
                
                ser.write(b'p')  # Unconditioned stimulus; send 'p' command to Arduino to trigger the puff
                # response = ser.readline().decode().strip()  # Read confirmation
                timestamp_response = cv2.getTickCount()    # Record when response is received, arduino code has completed execution
                
                time.sleep(0.05)  # Wait for air puff to complete
                
                # Do CS timestamp calculations after trial to avoid timing issues
                cs_timestamp = self.cs_tick_count / cv2.getTickFrequency() * 1000  # Convert to milliseconds
                # Do US timestamp calculations after trial to avoid timing issues
                us_timestamp = (timestamp_response / cv2.getTickFrequency()) * 1000
                self.df_stim.loc[len(self.df_stim)] = [i+1, cs_timestamp, us_timestamp]  # Add timestamps to df_stim
                
                print(f"Arduino response received at: {us_timestamp:.3f} ms")
                
                # Initiate ITI
                print("Waiting 10 seconds in between trials...")
                time.sleep(ITI)  # 10s ITI
                
                # End trial
                self.trial_in_progress = False
                
            except KeyboardInterrupt:
                print("Puff sequence interrupted by user.")
        
        self.running = False  # End experiment
        print("Experiment successfully completed.")
                
    def closeEvent(self, event):
        """Save csv's, close window and clean up
        """

        if self.fec_process:
            self.stop_fec_process()

        # Save FEC dataframe as csv
        fec_file = f"Code/IR Camera/capture/FEC/mouse_{mouse_id}_fec.csv"
        self.df_fec.to_csv(fec_file, index=False)
        
        # Save stimuli dataframe as csv
        stim_file = f"Code/IR Camera/capture/stim/mouse_{mouse_id}_stim.csv"
        self.df_stim.to_csv(stim_file, index=False)
        
        # Clean up
        self.running = False  # End experiment
        self.cap.release()  # Release camera
        ser.close()  # Close serial connection
        event.accept()
        
# Main application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
    

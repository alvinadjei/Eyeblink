import sys
import time
import serial
import threading
from multiprocessing import Event, Process, Queue
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
        self.df_fec = pd.DataFrame(columns=['Current Timestamp', 'Trial #', 'FEC'])  # Create df to hold FEC values
        
        # Dataframe to hold CS and US timestamps
        self.df_stim = pd.DataFrame(columns=['Trial #', 'CS Timestamp', 'US Timestamp'])  # Create df to hold CS and US timestamps for each trial
        
        # Multiprocessing
        # Initialize multiprocessing primitives
        self.running_event = Event()  # Used to signal if the FEC process is running
        self.trial_in_progress_event = Event()  # Used to signal if a trial is in progress
        self.fec_queue = Queue()  # Used for inter-process communication (passing FEC data)

    def start_fec_process(self):
        """Start the FEC acquisition process."""
        self.running_event.set()
        self.fec_process = Process(
            target=self.__fec_acquisition,
            args=(self.fec_queue, self.running_event, self.trial_in_progress_event, self.get_frame),
        )
        self.fec_process.start()

    def stop_fec_process(self):
        """Stop the FEC acquisition process."""
        self.running_event.clear()
        self.fec_process.join()
        self.fec_process = None
        
        # Retrieve all remaining FEC data from the queue
        fec_data = []
        while not self.fec_queue.empty():
            fec_data.append(self.fec_queue.get())

        # Add the data to the DataFrame in one batch
        if fec_data:
            df_new = pd.DataFrame(fec_data, columns=['Current Timestamp', 'Trial #', 'FEC'])
            self.df_fec = pd.concat([self.df_fec, df_new], ignore_index=True)
    
    def get_frame(self):
        """Retrieve the current processed frame for the FEC process."""
        return self.frame.copy() if self.frame is not None else None

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
        _, binary_dst = cv2.threshold(filtered_dst, binary_threshold, 255, cv2.THRESH_BINARY)  # Pixel values that are < binary_threshold are set to 0, and pixels that are > binary_threshold are set to 255
        
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

            # Display FEC value in the top left corner
            if self.light_fraction is not None:
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
    
    def __calculate_fec(self, frame):
        """Calculate the light fraction from the given frame."""
        
        # Ellipse dimensions
        start_x, start_y = self.ellipse_start
        end_x, end_y = self.ellipse_end

        center = ((start_x + end_x) // 2, (start_y + end_y) // 2)
        axes = (abs(end_x - start_x) // 2, abs(end_y - start_y) // 2)
        
        # Create a single-channel mask for the ellipse area
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Calculate the fraction of lighter pixels inside the ellipse using single-channel image
        masked_roi = cv2.bitwise_and(frame, frame, mask=mask)  # Apply the fixed mask to the current frame
        total_pixels = cv2.countNonZero(mask)  # Total pixels in the ellipse
        light_pixels = cv2.countNonZero(masked_roi)  # Light pixels within the ellipse
        self.light_fraction = (light_pixels / total_pixels) if total_pixels > 0 else 0  # Fraction of light pixels within the ellipse
        return self.light_fraction

    def __fec_acquisition(self, queue, running_event, trial_in_progress_event, frame_getter):
        """Function to run FEC acquisition in a separate process."""
        while running_event.is_set():
            if trial_in_progress_event.is_set():
                frame = frame_getter()
                if frame is not None:
                    fec_tick_count = cv2.getTickCount()
                    timestamp = fec_tick_count / cv2.getTickFrequency() * 1000  # Convert to milliseconds
                    #`calculate_fec()` is a helper function to compute the FEC from the frame
                    self.light_fraction = self.__calculate_fec(frame)
                    queue.put((timestamp, self.trial_ind, self.light_fraction))
            time.sleep(0.035)  # Approx 30 FPS

    def __start_delay_timer(self, delay_ms, callback):
        """Start a QTimer for the given delay, then execute the callback."""
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(callback)
        timer.start(delay_ms)
        
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
    
    def __stimuli(self):
        """Start the eyeblink experiment."""
        print("Starting experiment...")
        self.__run_trial()  # Begin the first trial

    def __run_trial(self):
        """Run a single trial."""
        if self.trial_ind >= num_trials or not self.running:
            print("Experiment ended early by user" if not self.running else "Experiment successfully completed.")
            self.running = False
            return  # Stop the experiment

        # Ensure stability before starting the trial
        try:
            self.__ensure_stability()  # Make sure the mouse eye is stable
        except Exception as e:
            print(f"Error ensuring eye stability: {e}")
            self.running = False
            return
    
    def __ensure_stability(self):
        """Check if FEC stays above 0.75 for at least 200 ms
        """
        start_time = None

        while self.running:
            if self.light_fraction < stability_threshold:  # If eye is >= 75% open (or < 25% closed)
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= stability_duration:  # Break loop if eyes stays open longer than 200 ms
                    print("Condition met: FEC stayed above 0.75 for at least 200 ms, commencing trial.")
                    break
            else:
                start_time = None  # Reset if the mouse blinks
        
        print(f"Trial {self.trial_ind + 1} in progress")
        self.trial_in_progress = True
        
        # Start recording 50 ms before conditioned stimulus
        self.__start_delay_timer(50, self.__start_conditioned_stimulus)

    def __start_conditioned_stimulus(self):
        """Start the conditioned stimulus and ISI."""
        self.__cond_stim()
        self.__start_delay_timer(int(ISI * 1000), self.__start_unconditioned_stimulus)

    def __start_unconditioned_stimulus(self):
        """Send the unconditioned stimulus."""
        try:
            ser.write(b'p')  # Trigger the air puff
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
            self.running = False
            return

        # Record response timestamp
        timestamp_response = cv2.getTickCount()

        # Perform CS and US timestamp calculations
        cs_timestamp = self.cs_tick_count / cv2.getTickFrequency() * 1000  # Convert to milliseconds
        us_timestamp = timestamp_response / cv2.getTickFrequency() * 1000  # Convert to milliseconds

        # Save the timestamps to the DataFrame
        self.df_stim.loc[len(self.df_stim)] = [self.trial_ind + 1, cs_timestamp, us_timestamp]
        print(f"CS at {cs_timestamp:.3f} ms, US at {us_timestamp:.3f} ms")

        # Wait 50 ms for the air puff to complete
        self.__start_delay_timer(50, self.__start_ITI)

    def __start_ITI(self):
        """Wait for the ITI and start the next trial."""
        print("Waiting 10 seconds between trials...")
        self.__start_delay_timer(ITI * 1000, self.__end_trial)

    def __end_trial(self):
        """End the current trial and start the next one."""
        self.trial_in_progress = False
        self.trial_ind += 1
        self.__run_trial()
    
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
    

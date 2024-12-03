import numpy as np
import time

import sounddevice as sd
import serial

# Initialize constants
arduino_port = '/dev/cu.usbserial-01C60315'  # Match this to Arduino's port, check by running 'ls /dev/cu.*' in terminal
baud_rate = 9600

# %% Conditioned Stimulus
def cond_stim():
    """Executes conditioned stimulus (plays the musical tone A5 for 300ms)
    """
    # Parameters for the tone
    frequency = 880.0  # Frequency in Hz (A5)
    duration = 0.3     # Duration in seconds
    sample_rate = 44100  # Sample rate in Hz

    # Generate the sound wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)  # 0.5 for volume control

    # Play the sound
    sd.play(waveform, samplerate=sample_rate)
    sd.wait()  # Wait until the sound has finished playing


# %% Main func
def main():
    n = 100
    
    # Establish serial connection
    ser = serial.Serial(arduino_port, baud_rate)
    time.sleep(2)  # Wait for the connection to establish
    
    for _ in range(n):
        try:
            cond_stim()  # Conditioned stimulus
            time.sleep(0.25)  # 250ms ISI
            ser.write(b'p')  # Unconditioned stimulus; send 'p' command to Arduino to trigger the puff
            time.sleep(10)  # 10s ITI
            
        except KeyboardInterrupt:
            print("Puff sequence interrupted by user.")
    
    # Close serial connection
    ser.close()
    

if __name__ == "__main__":
    main()
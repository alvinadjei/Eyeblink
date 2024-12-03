import numpy as np
import sounddevice as sd


# Parameters for the tone
frequency = 880.0  # Frequency in Hz (A440)
duration = 0.3    # Duration in seconds
sample_rate = 44100  # Sample rate in Hz

# Generate the sound wave
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
waveform = 0.5 * np.sin(2 * np.pi * frequency * t)  # 0.5 for volume control

# Play the sound
sd.play(waveform, samplerate=sample_rate)
sd.wait()  # Wait until the sound has finished playing


import cv2
import time

# Open the camera
camera = cv2.VideoCapture(0)  # Replace 0 with your camera's index if it's not the default one

# Set the desired resolution (1080p)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set the desired frame rate
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    print("Error: Unable to open the camera.")
    exit()

# Initialize variables for frame counting
num_frames = 30  # Number of frames to capture for testing
start_time = time.time()

# Capture frames
for i in range(num_frames):
    ret, frame = camera.read()
    if not ret:
        print(f"Error: Frame {i + 1} could not be read.")
        break

# Stop timing
end_time = time.time()

# Calculate fps
elapsed_time = end_time - start_time
fps = num_frames / elapsed_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
print(f"Calculated FPS: {fps:.2f}")

# Release the camera
camera.release()
cv2.destroyAllWindows()

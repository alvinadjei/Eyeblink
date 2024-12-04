# Test IR camera functionality

# import dependencies
import cv2
import numpy as np

# Open the file containing the camera's calibration vals
fs = cv2.FileStorage('Code/IR Camera/calibration/calib_params.xml', cv2.FILE_STORAGE_READ)

# Read the calibration vals and save as vars
mtx = fs.getNode("mtx").mat()
dist = fs.getNode("dist").mat()
rvecs = fs.getNode("rvecs").mat()
tvecs = fs.getNode("tvecs").mat()
fs.release()

# Initialize global variables
ellipse_params = None  # Parameters for the final ellipse
show_roi = False
eye_center = None  # Center of the ellipse

# Select ir webcam if plugged in
cap = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = cap.read()
    
    if ret:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray_frame.shape[:2]
        newcameramtx, dst_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(gray_frame, mtx, dist, None, newcameramtx)

        # crop the image (find better way to select roi)
        x, y, w, h = dst_roi
        dst = dst[y:y+h, x:x+w]
        
        # Apply 5 x 5 median filter to eliminate "salt and pepper" noise
        filtered_dst = cv2.medianBlur(dst, 5)
        
        # Apply a binary threshold to the grayscale image
        _, binary_dst = cv2.threshold(filtered_dst, 150, 255, cv2.THRESH_BINARY)  # Any pixel value below 150 will be set to 0, and above 150 will be set to 1 (255 for display)
        
        # Set roi ellipse when user presses the 's' key
        if cv2.waitKey(1) == ord('s'):
            show_roi = True
            
            # Define the fixed ROI ellipse
            eye_center = (w // 2, h // 2)  # Adjust as needed
            ellipse_axes = (int(w * 0.25), int(h * 0.2))  # Size of the ellipse
            ellipse_params = (eye_center, ellipse_axes, 0)  # Upright ellipse
        
        if show_roi and eye_center is not None:  # If the user has pressed the 's' key to establish the roi
            
            # # Adjust ellipse position using arrow keys
            # step_size = 100  # Pixels to move per key press
            # if cv2.waitKey(1) == 81:  # Left arrow key
            #     print('moved left')
            #     eye_center[0] = max(eye_center[0] - step_size, 0)
            # elif cv2.waitKey(1) == 82:  # Up arrow key
            #     eye_center[1] = max(eye_center[1] - step_size, 0)
            # elif cv2.waitKey(1) == 83:  # Right arrow key
            #     eye_center[0] = min(eye_center[0] + step_size, w - 1)
            # elif cv2.waitKey(1) == 84:  # Down arrow key
            #     eye_center[1] = min(eye_center[1] + step_size, h - 1)

            # # Update ellipse parameters with new center
            # ellipse_params = (tuple(eye_center), ellipse_axes, 0)
            
            # Show the ROI
            cv2.ellipse(binary_dst, ellipse_params, 150, 4)  # Grey outline for visualization
            
            # Create a mask for the elliptical ROI
            mask = np.zeros_like(binary_dst, dtype=np.uint8)
            cv2.ellipse(mask, ellipse_params, (255), -1)  # White-filled ellipse

            # Calculate the fraction of lighter pixels inside the ellipse
            masked_roi = cv2.bitwise_and(binary_dst, mask)  # Apply the fixed mask to the current frame
            total_pixels = cv2.countNonZero(mask)  # Total pixels in the ellipse
            light_pixels = cv2.countNonZero(masked_roi)  # Light pixels within the ellipse
            light_fraction = (light_pixels / total_pixels) if total_pixels > 0 else 0  # Fraction of light pixels within the ellipse

            cv2.putText(binary_dst, f"FEC: {light_fraction:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150), 2)
            
        # # Show processed frame
        # cv2.imshow('Processed Frame', binary_dst)
        # Show unprocessed frame
        cv2.imshow('Unprocessed Frame', dst)
            
        # Wait 1 millisecond, if we press 'q' key, break loop
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
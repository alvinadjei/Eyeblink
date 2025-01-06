import numpy as np
import cv2
import glob

# %% Setup

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
square_size = 26  # millimeters
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('calibration/samples/left' + '*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# %% Calibration

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  # returns the camera matrix, distortion coefficients, rotation and translation vectors

# save these parameters to be used in other files in the directory

# Create a FileStorage object
fs = cv2.FileStorage('calibration/calib_params.xml', cv2.FILE_STORAGE_WRITE)

# Write the arrays
fs.write("mtx", mtx)
fs.write("dist", dist)
fs.write("rvecs", np.array(rvecs))  # Needs to be a numpy array to write properly
fs.write("tvecs", np.array(tvecs))  # Needs to be a numpy array to write properly

# Release the FileStorage object
fs.release()
    
# %% Undistortion

img = cv2.imread('calibration/samples/left01.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibration/undistorted/calib_left01.png', dst)
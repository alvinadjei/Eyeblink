import cv2

# Open the file for reading
fs = cv2.FileStorage('Archived/Code/calibration/calib_params.xml', cv2.FILE_STORAGE_READ)

# Read the arrays
mtx = fs.getNode("mtx").mat()
dist = fs.getNode("dist").mat()
rvecs = fs.getNode("rvecs").mat()
tvecs = fs.getNode("tvecs").mat()

fs.release()

# %% Undistortion

# Sample image to be undistorted
samp = 'left05'

img = cv2.imread('calibration/samples/{}.jpg'.format(samp))
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibration/undistorted/calib_{}.png'.format(samp), dst)
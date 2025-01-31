# Test IR camera functionality

# import dependencies
import cv2

# Select ir webcam if plugged in
cap = cv2.VideoCapture(0)

i = 0
while True:
    # Capture frame
    ret, frame = cap.read()
    
    # Show captured frame
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Wait 1 millisecond, if we press 'c' key, save the current image
    if key == ord('c'):
        if i+1 < 10:  # if sample number is less than 10, make two digits
            cv2.imwrite('calibration/samples/left0{}.jpg'.format(i+1), frame)
        else:  # if sample number is 10 or greater, add as is
            cv2.imwrite('calibration/samples/left{}.jpg'.format(i+1), frame)
        i += 1  # increment i by one
    
    # Wait 1 millisecond, if we press 'q' key, break loop
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
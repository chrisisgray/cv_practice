import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print(cap.get(cv.CAP_PROP_FRAME_WIDTH),"x", cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    # Capture frame-by-frame
    # cap.read() returns a bool which we store in ret
    # if the frame is read correctly, then we can display it
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()
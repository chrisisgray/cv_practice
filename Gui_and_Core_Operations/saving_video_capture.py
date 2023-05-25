import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("can't receive frame")
        break

    # write the flipped frame
    out.write(frame)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
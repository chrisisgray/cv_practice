import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
assert cap is not None, "could not find a camera"

while(1):
    # read frame. frame comes in as BGR image
    _, frame = cap.read()

    # convert from BGR to HSV color-space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV in numpy array
    lower_blue = np.array([95, 30, 30])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 30, 30])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255]) 

    # threshold the HSV image to get only blue colors
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv.inRange(hsv, lower_green, upper_green)
    red_mask = cv.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    blue_res = cv.bitwise_and(frame, frame, mask = blue_mask)
    green_res = cv.bitwise_and(frame, frame, mask= green_mask)
    red_res = cv.bitwise_and(frame, frame, mask= red_mask)

    cv.imshow('frame', frame)
    cv.imshow('blue_mask', blue_mask)
    cv.imshow('blue res', blue_res)

    cv.imshow('green_mask', green_mask)
    cv.imshow('green_res', green_res)

    cv.imshow('red_mask', red_mask)
    cv.imshow('red res', red_res)

    k = cv.waitKey(5) & 0xFF

    if k == 27:
        break
    
cap.release()
cv.destroyAllWindows()
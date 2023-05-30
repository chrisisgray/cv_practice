import cv2 as cv
import numpy as np

# initialize camera object to read frames from
cap = cv.VideoCapture(0)

assert cap is not None, "Could not find camera"

while True:
    _, img = cap.read()

    img = cv.bilateralFilter(img, 9, 75, 75)
    img = cv.bilateralFilter(img, 9, 75, 75)
    img = cv.bilateralFilter(img, 9, 75, 75)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.Canny(gray, 200, 255)
    new_gray = gray

    cv.imshow('gray', gray)

    contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:

        M = cv.moments(c)
        if M['m00'] != 0 and M['m00'] > 50:
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(new_gray, (cX, cY), 3, (255, 255, 255), -1)



    cv.imshow('gray contours', new_gray)
    k = cv.waitKey(1000)
    if k == ord('q'):
        break
    cv.destroyAllWindows()

cap.release()
cv.destroyAllWindows()
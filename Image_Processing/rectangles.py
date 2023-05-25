import cv2 as cv
import numpy as np

img = cv.imread('heatsinks.jpg')
img = cv.resize(img, None, fx=0.2, fy=0.2)
img = cv.bilateralFilter(img, 9, 75, 75)
img = cv.bilateralFilter(img, 9, 75, 75)
img = cv.bilateralFilter(img, 9, 75, 75)


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_silver = np.array([91, 15, 20])  # silver in HSV
upper_silver = np.array([102, 120, 255])  # bright silver

silver_mask = cv.inRange(hsv, lower_silver, upper_silver)
kernel = np.ones((1,1), np.uint8)


opening = cv.morphologyEx(silver_mask, cv.MORPH_OPEN, kernel)


silver_res = cv.bitwise_and(img, img, mask = silver_mask)

contours, hierarchy = cv.findContours(opening, 1, 2)
print("Number of contours detected: ", len(contours))

for cnt in contours:
    x1,y1 = cnt[0][0]
    numSides = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
    if len(numSides) == 4:
        x,y,w,h = cv.boundingRect(cnt)
        if w * h > 50:
            print(x, y, w, h)
            img = cv.drawContours(img, [cnt], -1, (0,0,255), 3)
            cv.circle(img, (x+w, y+h), 2, (0, 0, 255), -1)

cv.imshow("Rectangles", img)
cv.imshow('silver mask', silver_mask)
# cv.imshow('opening mask', median)
cv.imshow('opening mask', opening)

cv.waitKey(0)
cv.destroyAllWindows()

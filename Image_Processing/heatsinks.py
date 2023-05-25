import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('heatsinks.jpg')
print(img.shape)
img = cv.resize(img, None, fx=0.2, fy=0.2)
print(img.shape)

img = cv.bilateralFilter(img, 9, 75, 75)
img = cv.bilateralFilter(img, 9, 75, 75)
img = cv.bilateralFilter(img, 9, 75, 75)

assert img is not None, "Could not find image"

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.Canny(gray, 200, 255)
# gray = cv.bilateralFilter(gray, 9, 75, 75)

cv.imshow('gray', gray)
lower_silver = np.array([91, 15, 20])  # silver in HSV
upper_silver = np.array([102, 120, 255])  # bright silver

silver_mask = cv.inRange(hsv, lower_silver, upper_silver)
silver_res = cv.bitwise_and(img, img, mask = silver_mask)

contours, hierarchy = cv.findContours(silver_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for c in contours:

    M = cv.moments(c)
    if M['m00'] != 0 and M['m00'] > 50:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(silver_res, (cX, cY), 3, (0,0, 255), -1)

cv.imshow("silver_res", silver_res)
cv.imshow('silver mask', silver_mask)
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])

plt.show()
cv.imshow('no red', img)
cv.waitKey(0)
cv.destroyAllWindows()
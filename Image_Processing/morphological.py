import cv2 as cv
import numpy as np

img = cv.imread('j.png')
assert img is not None, "Could not read image"

kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(img, kernel, iterations=1)
dilation = cv.dilate(img, kernel, iterations=3)


cv.imshow('erosion',erosion)
cv.imshow('dilation',dilation)

cv.waitKey(0)
cv.destroyAllWindows()
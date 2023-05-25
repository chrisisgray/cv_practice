import numpy as np
import cv2 as cv

img = cv.imread('roi.jpg')

assert img is not None, "file could not be read"

ball = img[280:340, 330:390]
img[280:333, 100:160] = ball
img[:,:,1] = 100

cv.imshow("new ball", img)

k = cv.waitKey(0)
cv.destroyAllWindows()
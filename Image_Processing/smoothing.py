import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('cards.jpg')

assert img is not None, "Could not find image!"

blur = cv.blur(img, (5, 5))
gblur = cv.GaussianBlur(img, (5,5), 0)
mblur = cv.medianBlur(img, 5)
bblur = cv.bilateralFilter(img, 9, 75, 75)

cv.imshow('original', img)
cv.imshow('blurred', blur)
cv.imshow('gblur', gblur)
cv.imshow('mblur', mblur)
cv.imshow('bblur', bblur)

cv.waitKey(0)
cv.destroyAllWindows()

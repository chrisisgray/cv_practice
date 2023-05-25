import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('cards.jpg')

if img is not None:
    img = cv.imread('..\Image_Processing\cards.jpg')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.medianBlur(img, 5)

ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                            cv.THRESH_BINARY, 11, 2)
th4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv.THRESH_BINARY, 11, 2)

ret, th3 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Otsu Binarization']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


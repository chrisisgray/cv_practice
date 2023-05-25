import cv2 as cv

img1 = cv.imread('bleach.jpg')
img2 = cv.imread('wpi-logo.png')

assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

print("Image 1 has dimensions", img1.shape)
print("Image 2 has dimensions", img2.shape)
e1 = cv.getTickCount()
# resize logo since it is too big
img2 = cv.resize(img2, (250, 250), cv.INTER_LINEAR)

# get the shape of the logo
rows, cols, channels = img2.shape

# the region of interest is the size of the logo.
# we plan on cutting these pixels out of the first image
# so we can place img2 in place of these
roi = img1[0:rows, 0:cols]

# convert the second image into gray
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imshow("img2gray", img2gray)

# create a binary mask containing only black and white image of logo
# cv.thrshold(src, min_threshold_value, max_threshold_value, thresholding type)
ret, mask = cv.threshold(img2gray, 0, 0, cv.THRESH_BINARY)
cv.imshow("mask", mask)


mask_inv = cv.bitwise_not(mask)
cv.imshow("mask_inv", mask_inv)

img1_bg = cv.bitwise_and(roi, roi,mask_inv)
cv.imshow("img1_bg", img1_bg)

img2_fg = cv.bitwise_and(img2, img2, mask)

dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
e2 = cv.getTickCount()

time = (e2 - e1) / cv.getTickFrequency()
print(time)

cv.imshow("res", img1)

cv.waitKey(0)
cv.destroyAllWindows()



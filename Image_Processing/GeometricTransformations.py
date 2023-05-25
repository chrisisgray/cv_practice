import cv2 as cv
import numpy as np

img = cv.imread('../Gui_and_Core_Operations/starry_night.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

assert img is not None, "could not find image"

# resize
# res = cv.resize(img, None,fx=2, fy=2, interpolation=cv.INTER_CUBIC)

rows, cols = img.shape

# wow, so the transformation matrix must be of type float
# M = np.array([[1, 0, 150], [0, 1, 150]], dtype=np.float32)

# rotating image 90 degrees about center with no scaling
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2), 180, 1)
res = cv.warpAffine(img, M, (cols, rows))

cv.imwrite('rotated_starry_night.jpg',res)
cv.imshow("res", res)
cv.waitKey(0)
cv.destroyAllWindows()
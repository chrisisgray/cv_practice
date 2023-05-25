# Standard imports
import cv2
import numpy as np

# Read image
im = cv2.imread("heatsinks.jpg")

im = cv2.resize(im, None, fx=0.2, fy=0.2)

hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
lower_silver = np.array([91, 15, 20])  # silver in HSV
upper_silver = np.array([102, 120, 255])  # bright silver

silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)
silver_res = cv2.bitwise_and(im, im, mask = silver_mask)

cv2.imshow("silver_res", silver_res)
cv2.imshow('silver mask', silver_mask)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 50

# Create a detector with the parameters
# OLD: detector = cv2.SimpleBlobDetector(params)
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(silver_mask)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(silver_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
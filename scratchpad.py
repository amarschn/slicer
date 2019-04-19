import numpy as np
import cv2

contours = np.array( [[4,4], [4,15], [15, 15], [15,4]] )
img = np.zeros( (20,20) ) # create a single channel 200x200 pixel black image
cv2.fillPoly(img, pts =[contours], color=(255,255,255))
cv2.imwrite('test.png', img)
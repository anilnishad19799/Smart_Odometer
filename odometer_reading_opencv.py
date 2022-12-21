import numpy as np
import cv2

# Load an color image in grayscale
# img = cv2.imread('shu4.jpg',0)

# cv2.imshow('image',img)
# cv2.imwrite('shu4_grey.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread('33.jpg',0)
# median = cv2.medianBlur(img,5)
# cv2.imshow('image',median)
# cv2.imwrite('33_grey_median.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('2.jpeg',0)
edges = cv2.Canny(img,100,200)

# cv2.imwrite('try_edges.jpg', edges)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


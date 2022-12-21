# try 1 

# import cv2
# import numpy as np

# img = cv2.imread('33.jpg', -1)

# rgb_planes = cv2.split(img)

# result_planes = []
# result_norm_planes = []
# for plane in rgb_planes:
#     dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#     bg_img = cv2.medianBlur(dilated_img, 21)
#     diff_img = 255 - cv2.absdiff(plane, bg_img)
#     norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     result_planes.append(diff_img)
#     result_norm_planes.append(norm_img)

# result = cv2.merge(result_planes)
# result_norm = cv2.merge(result_norm_planes)

# # cv2.imwrite('shadows_out.png', result)
# cv2.imwrite('33_try_1.png', result_norm)



# # Try 2
# import numpy as np
# import cv2

# # read an image with shadow...
# # and it converts to BGR color space automatically
# or_img = cv2.imread('33.jpg')

# # covert the BGR image to an YCbCr image
# y_cb_cr_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2YCrCb)

# # copy the image to create a binary mask later
# binary_mask = np.copy(y_cb_cr_img)

# # get mean value of the pixels in Y plane
# y_mean = np.mean(cv2.split(y_cb_cr_img)[0])

# # get standard deviation of channel in Y plane
# y_std = np.std(cv2.split(y_cb_cr_img)[0])

# # classify pixels as shadow and non-shadow pixels
# for i in range(y_cb_cr_img.shape[0]):
#     for j in range(y_cb_cr_img.shape[1]):

#         if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3):
#             # paint it white (shadow)
#             binary_mask[i, j] = [255, 255, 255]
#         else:
#             # paint it black (non-shadow)
#             binary_mask[i, j] = [0, 0, 0]

# # Using morphological operation
# # The misclassified pixels are
# # removed using dilation followed by erosion.
# kernel = np.ones((3, 3), np.uint8)
# erosion = cv2.erode(binary_mask, kernel, iterations=1)

# # sum of pixel intensities in the lit areas
# spi_la = 0

# # sum of pixel intensities in the shadow
# spi_s = 0

# # number of pixels in the lit areas
# n_la = 0

# # number of pixels in the shadow
# n_s = 0

# # get sum of pixel intensities in the lit areas
# # and sum of pixel intensities in the shadow
# for i in range(y_cb_cr_img.shape[0]):
#     for j in range(y_cb_cr_img.shape[1]):
#         if erosion[i, j, 0] == 0 and erosion[i, j, 1] == 0 and erosion[i, j, 2] == 0:
#             spi_la = spi_la + y_cb_cr_img[i, j, 0]
#             n_la += 1
#         else:
#             spi_s = spi_s + y_cb_cr_img[i, j, 0]
#             n_s += 1

# # get the average pixel intensities in the lit areas
# average_ld = spi_la / n_la

# # get the average pixel intensities in the shadow
# average_le = spi_s / n_s

# # difference of the pixel intensities in the shadow and lit areas
# i_diff = average_ld - average_le

# # get the ratio between average shadow pixels and average lit pixels
# ratio_as_al = average_ld / average_le

# # added these difference
# for i in range(y_cb_cr_img.shape[0]):
#     for j in range(y_cb_cr_img.shape[1]):
#         if erosion[i, j, 0] == 255 and erosion[i, j, 1] == 255 and erosion[i, j, 2] == 255:

#             y_cb_cr_img[i, j] = [y_cb_cr_img[i, j, 0] + i_diff, y_cb_cr_img[i, j, 1] + ratio_as_al,
#                                  y_cb_cr_img[i, j, 2] + ratio_as_al]

# # covert the YCbCr image to the BGR image
# final_image = cv2.cvtColor(y_cb_cr_img, cv2.COLOR_YCR_CB2BGR)

# cv2.imshow("im1", or_img)
# cv2.imshow("im2", final_image)
# cv2.imwrite('33_try_2.png', final_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# try 3
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread('33.jpg',0)

# # global thresholding
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# cv2.imwrite('33_try_3.png', th2)
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

# for i in range(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()


# try 4
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('3.jpeg',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

cv2.imwrite('3_try_4.png', th3)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]


for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
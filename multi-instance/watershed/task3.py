# 9th Nov
# created by Liqi Jiang

# transform the img from rbg to hsv
# obtain the green part   ------- lower_green = np.array([35, 43, 46])  upper_green = np.array([77, 255, 255])
# transform to gray image
# use threshold and median filter
# erode and dilate the image ---- remove the black noisy points
# compute the distance -> use threshold
# draw contours -> markers
# watershed -> marker
# bitwise_not operation -> mark
# watershed ------  labels_watershed = watershed(-distance, mark, mask=dilated)

#######  ！！！ Bad performance ---- an image of a large mass of adhesions

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


image = cv2.imread("Plant/Ara2013-Canon/ara2013_plant001_rgb.png")
img = cv2.medianBlur(image, 3)
# plt.imshow(img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 43, 46])
upper_green = np.array([77, 255, 255])
mask = cv2.inRange(img_hsv, lower_green, upper_green)
img_green = cv2.bitwise_and(img, img, mask=mask)
# img_green = cv2.cvtColor(res, cv2.THRESH_BINARY)

#kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
#img_filter2d = cv2.filter2D(img_green, ddepth=-1, kernel=kernel)
#img_result = img_green - img_filter2d
#plt.imshow(img_result)
#plt.show()

img_gray = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)
img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img_thresh = cv2.medianBlur(img_thresh, 5)
#plt.title("img_thres")
#plt.imshow(img_thresh, cmap='gray')
#plt.show()

kernel=np.uint8(np.zeros((5, 5)))
for x in range(5):
    kernel[x,2] = 1
    kernel[2,x] = 1
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
eroded = cv2.erode(img_thresh, kernel)
dilated = cv2.dilate(eroded, kernel)

# plt.title("sobel_x")
# plt.imshow(sobel_x, cmap='gray')
# plt.show()
# plt.title("sobel_y")
# plt.imshow(sobel_y, cmap='gray')
# plt.show()

sobel_x = cv2.Sobel(dilated, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(dilated, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
dilated = (dilated + img_sobel).astype('uint8')
plt.title("sobel")
plt.imshow(dilated, cmap='gray')
plt.show()


dilated = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# plt.axis('off')
# plt.title("dilated")
# plt.imshow(dilated, cmap='gray')
# plt.show()

# img_thres = cv2.bitwise_and(img_thresh, img_thresh, mask=img_sobel)
# plt.title("img_thres")
# plt.imshow(img_thres, cmap='gray')
# plt.show()


distance = ndi.distance_transform_edt(dilated)
cv2.normalize(distance, distance, norm_type=cv2.NORM_MINMAX)
#plt.title("distance")
#plt.imshow(-distance, cmap='gray')
#plt.show()

distance_binary = cv2.threshold(distance, 0.4, 1, cv2.THRESH_BINARY)[1].astype('uint8')
# plt.title("distance binary")
# plt.imshow(distance_binary, cmap='gray')
# plt.show()
# distance_binary = cv2.cvtColor(distance_binary, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(distance_binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
markers = np.zeros((img_gray.shape[0], img_gray.shape[1]), dtype='float32')
for i in range(len(contours)):
    cv2.drawContours(markers, contours, i, i+1, -1)
# plt.title("markers")
# plt.imshow(markers)
# plt.show()

kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3)).astype('uint8')
img = cv2.morphologyEx(img_green, cv2.MORPH_ERODE, kernel).astype('uint8')
# plt.title("morphologyEx")
# plt.imshow(img)
# plt.show()

marker = cv2.watershed(img, markers.astype('int32'))
mark = np.zeros((markers.shape[0], markers.shape[1]), dtype='uint8')
mark = markers.astype('uint8')
cv2.bitwise_not(mark)
# plt.title("watershed")
# plt.imshow(mark)
# plt.show()
# circle(markers, Point(5, 5), 3, Scalar(255), -1);
# print(type(contours))
# print(distance1)
# local_maxi = peak_local_max(distance, indices=False, min_distance=20,labels=img_gray)
# print(local_maxi1)
# markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
labels_watershed = watershed(-distance, mark, mask=dilated)
plt.figure(3, figsize=(20,10))
plt.subplot(1,3,1)
plt.title("original")
plt.axis('off')
plt.imshow(image)
plt.subplot(1,3,2)
plt.axis('off')
plt.title("dilated")
plt.imshow(dilated, cmap='gray')
plt.subplot(1,3,3)
plt.axis('off')
plt.title("segmented")
plt.imshow(labels_watershed, cmap='nipy_spectral')
fname = "ara2013_plant001_result.png"
plt.savefig(fname)
plt.show()

















# 10th Nov
# created by Liqi Jiang

# transform the img from rbg to hsv
# obtain the green part   ------- lower_green = np.array([35, 43, 46])  upper_green = np.array([77, 255, 255])
# transform to gray image
# use threshold and median filter
# erode and dilate the image ---- remove the black noisy points
# watershed ----  compute the distance, markers, labels

###### !!! better than task3.py, but overfitting


import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def Dice(inlabel, gtlabel, i, j):
    # calculate Dice score for the given labels i and j.
    inMask = (inlabel == i)
    gtMask = (gtlabel == j)
    insize = np.sum(inMask)
    gtsize = np.sum(gtMask)
    overlap = np.sum(inMask & gtMask)
    return 2 * overlap / float(insize + gtsize)


def BestDice(inlabel, gtlabel):
    '''
    inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    score: Dice score
    '''
    if (inlabel.shape == gtlabel.shape):
        maxInLabel = int(np.max(inlabel))
        minInLabel = int(np.min(inlabel))
        maxGtLabel = int(np.max(gtlabel))
        minGtLabel = int(np.min(gtlabel))
        score = 0  # initialize output

        # loop all labels of inLabel.
        for i in range(minInLabel, maxInLabel + 1):
            sMax = 0
            # loop all labels of gtLabel.
            for j in range(minGtLabel, maxGtLabel + 1):
                s = Dice(inlabel, gtlabel, i, j)
                # keep max Dice value for label i.
                if sMax < s:
                    sMax = s
            score += sMax  # sum up best found values.
        score = score / float(maxInLabel - minInLabel + 1)
        return score
    else:
        return 0


def best_dice(l_a, l_b):
    """
    Best Dice function
    :param l_a: list of binary instances masks
    :param l_b: list of binary instances masks
    :return: best dice estimation
    """
    result = 0
    for a in l_a:
        best_iter = 0
        for b in l_b:
            inter = 2 * float(np.sum(a * b)) / float(np.sum(a) + np.sum(b))
            if inter > best_iter:
                best_iter = inter
        result += best_iter
    if 0 == len(l_a):
        return 0

    return result / len(l_a)


def symmetric_best_dice(l_ar, l_gr):
    """
    Symmetric Best Dice function
    :param l_ar: list of output binary instances masks
    :param l_gr: list of binary ground truth masks
    :return: Symmetric best dice estimation
    """
    return np.max([best_dice(l_ar, l_gr), best_dice(l_gr, l_ar)])



def get_as_list(array):
    """
    Convert indexes to list
    """
    objects = []
    pixels = np.unique(array)
    for l, v in enumerate(pixels[1:]):
        bin_mask = np.zeros_like(array)
        bin_mask[array == v] = 1
        objects.append(bin_mask)
    return objects


def sdb_score(rgb, label):
    image = cv2.imread(rgb)
    img = cv2.medianBlur(image, 3)
    # plt.imshow(img)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    img_green = cv2.bitwise_and(img, img, mask=mask)
    # img_green = cv2.cvtColor(res, cv2.THRESH_BINARY)
    # plt.axis('off')
    # plt.imshow(img_green)
    # plt.show()

    img_gray = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)

    img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_thresh = cv2.medianBlur(img_thresh, 5)

    kernel=np.uint8(np.zeros((5, 5)))
    for x in range(5):
        kernel[x,2] = 1
        kernel[2,x] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilated = cv2.erode(img_thresh, kernel)
    # plt.imshow(dilated, cmap='gray')
    # plt.show()
    '''
    sobel_x = cv2.Sobel(dilated, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(dilated, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    dilated = (dilated + img_sobel).astype('uint8')
    plt.title("sobel")
    plt.imshow(dilated, cmap='gray')
    plt.show()
    '''


    # dilated = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    '''
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(4, 4))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    dilated = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    plt.imshow(dilated, cmap='gray')
    plt.show()
    '''

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(dilated)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((4, 4)),
                            labels=dilated)
    markers = ndi.label(local_maxi)[0]
    # plt.imshow(markers, cmap='gray')
    # plt.show()
    labels = watershed(-distance, markers, mask=dilated)
    # labels = mask_to_rgb(labels)
    labels = get_as_list(labels)
    # rgb_labels = cv2.cvtColor(labels, cv2.COLOR_GRAY2RGB)
    # plt.imshow(labels, cmap='gray')
    # plt.show()
    # print(len(labels))
    ground_truth = cv2.imread(label, 0)
    # ground_truth = ground_truth[:ground_truth.shape[0], :ground_truth.shape[1]]
    # print(ground_truth.shape)
    # ground_truth = cv2.threshold(ground_truth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    ground_truth = get_as_list(ground_truth)
    # print(len(ground_truth))
    # ground_truth = mask_to_rgb(ground_truth)
    # plt.imshow(ground_truth, cmap='gray')
    # plt.show()
    score = symmetric_best_dice(labels, ground_truth)
    # print(score)
    return score


# sbd = sdb_score("Plant/Ara2012/ara2012_plant001_rgb.png","Plant/Ara2012/ara2012_plant001_label.png")
# print(sbd)




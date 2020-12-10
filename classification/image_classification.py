import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# SIFT
def getFeature(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # print(descriptors)
    return descriptors


# use k-means to cluster sift features to 50 classes
def get_feature_bag(features):
    wordCnt = 50
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    # get initial center point of k-means
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(features, wordCnt, None, criteria, 20, flags)
    return centers


def getFeatureVector(descriptor, feature_bag):
    featureVec = np.zeros((1, 50))
    for i in range(0, descriptor.shape[0]):
        fi = descriptor[i]
        diffMat = np.tile(fi, (50, 1)) - feature_bag
        # axis = 1, caulate the distance from feature to each center point
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        # get the smallest distance --- nearest center point
        idx = sortedIndices[0]
        featureVec[0][idx] += 1
    return featureVec


if __name__ == "__main__":
    arabidopsis_path = "Plant/Ara2013-Canon/"
    tobacco_path = "Plant/Tobacco/"
    arabidopsis = []
    tobacco = []
    for dirpath, dirnames, filenames in os.walk(arabidopsis_path):
        for file in filenames:
            if file.endswith("_rgb.png"):
                arabidopsis.append(arabidopsis_path+file)
    for dirpath, dirnames, filenames in os.walk(tobacco_path):
        for file in filenames:
            if file.endswith("_rgb.png"):
                tobacco.append(tobacco_path+file)
    # print(arabidopsis)
    # print(tobacco)
    # print(len(arabidopsis))
    # print(len(tobacco))

    training_set_size = 0.2
    x_train_size_ara = int(len(arabidopsis)*training_set_size)
    x_train_size_tob = int(len(tobacco)*training_set_size)

    # get feature descriptor of trainging images
    features = np.float32([]).reshape(0,128)    # store the feature descriptor of trainging images
    for file in arabidopsis[0:x_train_size_ara]:
        img_gray = cv2.imread(file, 0)
        # print(img_gray.shape)
        # preprocess
        img_gray = cv2.resize(img_gray, (100, int(100*img_gray.shape[0]/img_gray.shape[1])))
        descriptor = getFeature(img_gray)  # obtain sift descriptor
        # print(img_des.shape)
        features = np.append(features, descriptor, axis=0)
    for file in tobacco[0:x_train_size_tob]:
        img_gray = cv2.imread(file, 0)
        # print(img_gray.shape)
        # preprocess
        img_gray = cv2.resize(img_gray, (100, int(100*img_gray.shape[0]/img_gray.shape[1])))
        descriptor = getFeature(img_gray)  # obtain sift descriptor
        features = np.append(features, descriptor, axis=0)

    # bag of feature
    feature_bag = get_feature_bag(features)
    #print(feature_bag.shape)

    # calculate feature vector
    feature_vector = np.float32([]).reshape(0,50)
    x_train = np.float32([]).reshape(0,50)
    x_test = np.float32([]).reshape(0,50)
    labels = np.float32([])
    y_train = np.float32([])
    y_test = np.float32([])
    idx = 0
    for file in arabidopsis:
        img_gray = cv2.imread(file, 0)
        img_gray = cv2.resize(img_gray, (100, int(100 * img_gray.shape[0] / img_gray.shape[1])))
        descriptor = getFeature(img_gray)
        img_vec = getFeatureVector(descriptor, feature_bag)
        #feature_vector = np.append(feature_vector, img_vec, axis=0)
        #labels = np.append(labels, 0)
        if idx < x_train_size_ara:
            x_train = np.append(x_train, img_vec, axis=0)
            y_train = np.append(y_train, 0)
        else:
            x_test = np.append(x_test, img_vec, axis=0)
            y_test = np.append(y_test, 0)
        idx += 1
    # print(x_train.shape)
    idx = 0
    for file in tobacco:
        img_gray = cv2.imread(file, 0)
        img_gray = cv2.resize(img_gray, (100, int(100 * img_gray.shape[0] / img_gray.shape[1])))
        descriptor = getFeature(img_gray)
        img_vec = getFeatureVector(descriptor, feature_bag)
        # feature_vector = np.append(feature_vector, img_vec, axis=0)
        # labels = np.append(labels, 1)
        if idx < x_train_size_tob:
            x_train = np.append(x_train, img_vec, axis=0)
            y_train = np.append(y_train, 1)
        else:
            x_test = np.append(x_test, img_vec, axis=0)
            y_test = np.append(y_test, 1)
        idx += 1

    # print(x_test.shape)
    # print(x_train_size_ara)
    # print(x_train_size_tob)

    # print(data)
    # print(data[0])
    # print(len(arabidopsis)+len(tobacco))
    # print(len(data[5]))
    # generate the training set and test set
    # x_train, x_test, y_train, y_test = train_test_split(feature_vector, labels, test_size=0.3, random_state=0)


    # pattern recognition
    # SVM
    '''
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    print(classification_report(y_test, y_pred))
    '''
    #img1 = cv2.imread('tobacco_plant003_rgb.png')
    #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #sift = cv2.SIFT_create()
    #keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    #img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
    #plt.imshow(img_1)
    #plt.show()
    '''
    # find the best number of neighbors
    k_range = range(1, 26)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        scores.append(accuracy_score(y_test, y_pred))

    print(scores)
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()
    '''
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(classification_report(y_test, y_pred))




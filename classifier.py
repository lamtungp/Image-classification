import cv2
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Local dependencies

import constants
import descriptors
import filenames
import utils


class Classifier:
    """
    Class for making training and testing in image classification.
    """

    def __init__(self, dataset, log):
        self.dataset = dataset
        self.log = log

    def train(self, knn_kernel, k, des_name):
        isTrain = True
        des_name = constants.SIFT_FEAT_NAME
        x_filename = filenames.vlads_train(k, des_name)
        print("Getting global descriptors for the training set.")
        start = time.time()
        x, y, cluster_model = self.get_data_and_labels(
            self.dataset.get_train_set(), None, k, des_name, isTrain)
        utils.save(x_filename, x)
        print("cluster_model", cluster_model)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        knn_filename = filenames.knn(k, des_name, knn_kernel)
        print("Elapsed time getting descriptors {0}".format(elapsed_time))
        print("Calculating the Support Vector Machine for the training set...")
        clf = KNeighborsClassifier()
        clf.fit(x, y)
        return clf, cluster_model

    def test(self, knn, cluster_model, k):
        isTrain = False
        des_name = constants.SIFT_FEAT_NAME
        print("Getting global descriptors for the testing set...")
        start = time.time()
        x, y, cluster_model = self.get_data_and_labels(
            self.dataset.get_test_set(), cluster_model, k, des_name, isTrain)
        end = time.time()
        start = time.time()
        result = knn.predict(x)
        end = time.time()
        self.log.predict_time(end - start)
        mask = result == y
        correct = np.count_nonzero(mask)
        accuracy = (correct * 100.0 / result.size)
        self.log.accuracy(accuracy)
        return result, y, mask

    def get_data_and_labels(self, img_set, cluster_model, k, des_name, isTrain):
        y = []
        x = None
        img_descs = []

        for class_number in range(len(img_set)):
            img_paths = img_set[class_number]
            step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
            for i in range(len(img_paths)):
                if (step > 0) and (i % step == 0):
                    percentage = (100 * i) / len(img_paths)
                img = cv2.imread(img_paths[i])
                des, y = descriptors.sift(img, img_descs, y, class_number)
        isTrain = int(isTrain)
        if isTrain == 1:
            X, cluster_model = descriptors.cluster_features(
                des, cluster_model=MiniBatchKMeans(n_clusters=k))
        else:
            X = descriptors.img_to_vect(des, cluster_model)
        print('X', X.shape, X)
        y = np.float32(y)[:, np.newaxis]
        x = np.matrix(X)
        return x, y, cluster_model

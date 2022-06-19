import cv2
import numpy as np
import time
import os

# Local dependencies
from classifier import Classifier
from dataset import Dataset
import constants
import utils
import filenames
from log import Log


def main(k=10, knn_kernel=cv2.ml.KNearest_create()):
    # Check for the dataset of images
    if not os.path.exists(constants.DATASET_PATH):
        print("Dataset not found, please copy one.")
        return
    dataset = Dataset(constants.DATASET_PATH)
    dataset.generate_sets()
    # Check for the directory where stores generated files
    if not os.path.exists(constants.FILES_DIR_NAME):
        os.makedirs(constants.FILES_DIR_NAME)

    des_name = constants.SIFT_FEAT_NAME
    log = Log(k, des_name, knn_kernel)

    codebook_filename = filenames.codebook(k, des_name)
    print('codebook_filename')
    print(codebook_filename)
    start = time.time()
    end = time.time()
    log.train_des_time(end - start)
    start = time.time()
    end = time.time()
    log.codebook_time(end - start)
    # Train and test the dataset
    classifier = Classifier(dataset, log)
    knn, cluster_model = classifier.train(
        knn_kernel, k, des_name)
    print("Training ready. Now beginning with testing")
    result, labels, mask = classifier.test(
        knn, cluster_model, k)
    print('test result')
    # Store the results from the test
    classes = dataset.get_classes()
    log.classes(classes)
    log.classes_counts(dataset.get_classes_counts())
    result_filename = filenames.result(k, des_name, knn_kernel)
    test_count = len(dataset.get_test_set()[0])
    # for j in range(len(labels)):
    #     print(j)
    result_label = []

    for i in result:
        if (i == 0.):
            result_label.append('caybang')
        if (i == 1.):
            result_label.append('caychuoi')
        if (i == 2.):
            result_label.append('cayco')
        if (i == 3.):
            result_label.append('caydao')
        if (i == 4.):
            result_label.append('dinhlang')
        if (i == 5.):
            result_label.append('caymai')
        if (i == 6.):
            result_label.append('cayphong')
        if (i == 7.):
            result_label.append('cayquat')
        if (i == 8.):
            result_label.append('cayphuong')
        if (i == 9.):
            result_label.append('hoagiay')

    print(result_label)
    result_matrix = np.reshape(result_label, (test_count, len(classes)))
    utils.save_csv(result_filename, result_matrix)

    # Create a confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.uint32)
    for i in range(len(result)):
        predicted_id = int(result[i])
        real_id = int(labels[i])
        confusion_matrix[real_id][predicted_id] += 1

    print("Confusion Matrix =\n{0}".format(confusion_matrix))
    log.confusion_matrix(confusion_matrix)
    log.save()
    print("Log saved on {0}.".format(filenames.log(k, des_name, knn_kernel)))
    # Show a plot of the confusion matrix on interactive mode
    print(result_filename)
    # utils.show_conf_mat(confusion_matrix)
    #raw_input("Press [Enter] to exit ...")


if __name__ == '__main__':
    main()

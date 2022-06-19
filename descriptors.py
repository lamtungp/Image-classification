import cv2
import numpy as np

def sift(img, img_descs, y, class_number):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    if des is not None:
        img_descs.append(des)
        y.append(class_number)
    else:
        print('Found you!!!!!!!')
    return img_descs, y


def gen_codebook(dataset, descriptors, k=10):
    k = int(k)
    print(type(dataset), type(descriptors), type(k))

    iterations = 10
    epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    compactness, labels, centers = cv2.kmeans(
        descriptors, k, None, criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)
    return centers


def img_to_vect(img_descs, cluster_model):
    clustered_descs = [cluster_model.predict(
        img_desc) for img_desc in img_descs]

    img_bow_hist = np.array([np.bincount(
        clustered_desc, minlength=cluster_model.n_clusters) for clustered_desc in clustered_descs])
    return img_bow_hist


def cluster_features(img_descs, cluster_model):
    n_clusters = cluster_model.n_clusters
    # Concatenate all descriptors in the training set together
    training_descs = img_descs
    all_train_descriptors = [
        desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    if all_train_descriptors.shape[1] != 128:
        raise ValueError(
            'Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(
        raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print('done generating BoW histograms.')

    return X, cluster_model

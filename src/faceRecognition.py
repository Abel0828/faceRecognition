# -*- coding: UTF-8 -*-
import math
import numpy as np
import cv2

import tensorflow as tf
import facenet

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class FaceRecognition(object):
    def __init__(self, recognition_model, gpu_memory_fraction=0):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.image_size = 160
        self.batch_size = 50

        self.recognition_model = recognition_model

        # graph and session:
        self.recognition_graph = None
        self.recognition_sess = None

        # faceRecognition embedding
        self.images_recognition_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.embedding_size = None

        # classifier
        self.svm_model = None
        self.knn_model = None
        self.n_neighbors = 1

        # store result
        self.face_data = []  # [imageObject(name, path), ...]
        self.class_names = {}  # {'0': Bob, '1': Peter, ...}
        self.labels = []  # [0, 0, 1, 2, 2, 2, 3, ...]
        self.face_embeddings = np.array([])  # np.array([[], []])  128 vectors

        self.init_config()
        return

    def __del__(self):
        """
            frees all resources associated with the session
        """
        self.recognition_sess.close()
        return

    def init_config(self):
        """
            initial configuration
        :return:
        """
        self.load_extract_feature_model()

        return

    def update_model_data(self, data_folder):
        """
        update the data for the model
        :param data_folder:
        :return:
        """
        self.face_data, self.class_names = facenet.get_dataset(data_folder)
        self.load_embeddings(self.face_data)

        # train classifier
        self.train_svm_model(self.face_embeddings, self.labels)
        self.train_knn_model(self.face_embeddings, self.labels)
        return

    ################################
    # Face faceRecognition
    ################################
    def load_extract_feature_model(self):
        """load feature extraction model, transfer an image to a 128 dimension vector."""
        print('Loading feature extraction model')
        self.recognition_graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        self.recognition_sess = tf.Session(graph=self.recognition_graph,
                                           config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with self.recognition_graph.as_default():
            print('recognition_sess: ', self.recognition_sess)
            with self.recognition_sess.as_default():
                print('self.pretrained_model: ', self.recognition_model)
                facenet.load_model(self.recognition_model)
                # Get input and output tensors
                self.images_recognition_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]
        return

    def load_embeddings(self, dataset):
        """
            calculate 128 dimension embeddings of faceData
        :param dataset: list of images object, such as self.face_data
        :return:
        """
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        paths, self.labels = facenet.get_image_paths_and_labels(dataset)
        # images = facenet.load_data(paths, False, False, self.image_size)
        # feed_dict = {self.images_recognition_placeholder: images, self.phase_train_placeholder: False}
        # self.face_embeddings = self.recognition_sess.run(self.embeddings, feed_dict=feed_dict)
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            feed_dict = {self.images_recognition_placeholder: images, self.phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.recognition_sess.run(self.embeddings, feed_dict=feed_dict)
        self.face_embeddings = emb_array
        return

    def calculate_embeddings(self, images):
        """
            calculate 128 dimension embeddings of detected images
        :param images: list of images, np.array([[image], [image], ...]), pre-whiten image, 160*160, RGB
        :return: embeddings of detected images (160*160, after crop)
        """
        feed_dict = {self.images_recognition_placeholder: images, self.phase_train_placeholder: False}
        tmp_embeddings = self.recognition_sess.run(self.embeddings, feed_dict=feed_dict)
        return tmp_embeddings

    def train_svm_model(self, X, y):
        """
        :param X: image embedding, X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        :param y: image label, y = np.array([1, 1, 2, 2])
        :return:
        """
        self.svm_model = SVC(kernel='linear', probability=True)
        self.svm_model.fit(X, y)
        return

    def train_knn_model(self, X, y):
        """
        :param X: image embedding, X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        :param y: image label, y = np.array([1, 1, 2, 2])
        :return:
        """
        self.knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn_model.fit(X, y)
        return

    def recognize_face_svm(self, image_embeddings):
        """
        use svm algorithm to classify image
        :param image_embeddings: list of image embeddings, np.array([[], [], ...])
        :return face_class_prob: list of face distance and label [[class, prob], [], ...]
        """
        predictions = self.svm_model.predict_proba(image_embeddings)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        face_class_prob = []
        for i in range(len(best_class_indices)):
            face_class_prob.append([self.class_names[best_class_indices[i]], best_class_probabilities[i]])
        return face_class_prob

    def recognize_face_knn(self, image_embeddings):
        """
        use knn algorithm to classify image
        :param image_embeddings: list of image embeddings, np.array([[], [], ...])
        :return face_class_prob: list of face distance and label [[class, prob], [], ...]
        """
        predictions = self.knn_model.predict_proba(image_embeddings)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        face_class_prob = []
        for i in range(len(best_class_indices)):
            face_class_prob.append([self.class_names[best_class_indices[i]], best_class_probabilities[i]])
        return face_class_prob


if __name__ == '__main__':
    face_data_folder = '../../data/faceData/'
    recognition_model = '../../data/models/recognition/20170512-110547/20170512-110547.pb'
    faceRecognition = FaceRecognition(face_data_folder, recognition_model)

    imgPath = '../../data/test/60_2.png'
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('image.shape: ', image.shape)

    print('len(face_data): ', len(faceRecognition.face_data))
    print('len(labels): ', len(faceRecognition.labels))
    print('class_names): ', len(faceRecognition.class_names))
    print('face_embeddings.shape: ', faceRecognition.face_embeddings.shape)

    imageList = np.array([image])

    tmp_embeddings = faceRecognition.calculate_embeddings(imageList)
    print('tmp_embbedings.shape: ', tmp_embeddings.shape)
    print('svm: ', faceRecognition.recognize_face_svm(tmp_embeddings))
    print('knn: ', faceRecognition.recognize_face_knn(tmp_embeddings))




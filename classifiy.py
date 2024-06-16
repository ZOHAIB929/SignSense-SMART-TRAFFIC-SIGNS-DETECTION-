import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
# Local modules
from common import clock, mosaic

# Parameters
SIZE = 32
NUM_CLASSES = 13


def load_traffic_dataset():
    dataset = []
    labels = []
    for class_id in range(NUM_CLASSES):
        image_files = listdir(f"./dataset/{class_id}")
        for file in image_files:
            if file.endswith('.png'):
                path = f"./dataset/{class_id}/{file}"
                print(path)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (SIZE, SIZE))
                dataset.append(img)
                labels.append(class_id)
    return np.array(dataset), np.array(labels)


def deskew(img):
    moments = cv2.moments(img)
    if abs(moments['mu02']) < 1e-2:
        return img.copy()
    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


class StatModel:
    def load(self, filename):
        self.model.load(filename)

    def save(self, filename):
        self.model.save(filename)


class SVM(StatModel):
    def __init__(self, C=12.5, gamma=0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, data, samples, labels):
    predictions = model.predict(samples)
    error_rate = (labels != predictions).mean()
    print(f'Accuracy: {(1 - error_rate) * 100:.2f} %')

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int32)
    for true_label, pred_label in zip(labels, predictions):
        confusion_matrix[int(true_label), int(pred_label)] += 1
    print('Confusion matrix:')
    print(confusion_matrix)

    visualization = []
    for img, correct in zip(data, predictions == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not correct:
            img[..., :2] = 0
        visualization.append(img)
    return mosaic(16, visualization)


def preprocess_data(data):
    return np.float32(data).reshape(-1, SIZE * SIZE) / 255.0


def get_hog_descriptor():
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    return hog


def train_model():
    print('Loading dataset...')
    data, labels = load_traffic_dataset()
    print(data.shape)

    print('Shuffling dataset...')
    rand_state = np.random.RandomState(10)
    shuffle_indices = rand_state.permutation(len(data))
    data, labels = data[shuffle_indices], labels[shuffle_indices]

    print('Deskewing images...')
    deskewed_data = list(map(deskew, data))

    print('Setting up HoG descriptor...')
    hog = get_hog_descriptor()

    print('Computing HoG descriptors...')
    hog_descriptors = [hog.compute(img) for img in deskewed_data]
    hog_descriptors = np.squeeze(hog_descriptors)

    print('Splitting data into training and test sets...')
    train_count = int(0.9 * len(hog_descriptors))
    train_data, test_data = np.split(deskewed_data, [train_count])
    train_descriptors, test_descriptors = np.split(hog_descriptors, [train_count])
    train_labels, test_labels = np.split(labels, [train_count])

    print('Training SVM model...')
    model = SVM()
    model.train(train_descriptors, train_labels)

    print('Saving SVM model...')
    model.save('svm_model.dat')
    return model


def get_label(model, img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (SIZE, SIZE))
    deskewed_img = deskew(resized_img)
    hog = get_hog_descriptor()
    hog_descriptor = hog.compute(deskewed_img).reshape(1, -1)
    return int(model.predict(hog_descriptor)[0])

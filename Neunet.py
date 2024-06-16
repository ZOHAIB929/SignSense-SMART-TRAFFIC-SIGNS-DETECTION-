from __future__ import generators, generator_stop

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from skimage import transform, exposure, io
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
from collections import OrderedDict
from tqdm import tqdm
import util  # Assuming `util` is a module for tensor conversion
import keras

class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        # First Conv -> ReLU -> BatchNorm -> MaxPool
        model.add(Conv2D(8, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second set of layers
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third set of layers
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Fully Connected layers and softmax classifier
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

def set_params():
    global num_epochs, init_lr, bs
    # Hyperparameters
    num_epochs = 30
    init_lr = 1e-3
    bs = 64

def image_augmentation():
    global aug
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest"
    )

def train():
    opt = Adam(learning_rate=init_lr, decay=init_lr / (num_epochs * 0.5))
    model = TrafficSignNet.build(width=32, height=32, depth=3, classes=num_labels)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    H = model.fit(
        aug.flow(trainX, trainY, batch_size=bs),
        validation_data=(testX, testY),
        steps_per_epoch=trainX.shape[0] // bs,
        epochs=num_epochs,
        class_weight=class_weight,
        verbose=1
    )

def load_model(path):
    return keras.models.load_model(path)

SIGNS = [
    'ERROR',
    'STOP',
    'TURN LEFT',
    'TURN RIGHT',
    'DO NOT TURN LEFT',
    'DO NOT TURN RIGHT',
    'ONE WAY',
    'SPEED LIMIT',
    'OTHER'
]

def filter_predictions():
    for sign in SIGNS:
        for label in range(num_labels):
            gtr = trainX[label]
            set1 = set(map(str.lower, sign.split()))
            set2 = set(map(str.lower, gtr.split()))
            if set1 == set2:
                continue

def train_neural_network(opt):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    for data in tqdm(dataset):
        minibatch = 1
        reset = model.inference(data['label'], data['inst'])

        visuals = OrderedDict([
            ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
            ('reset_image', util.tensor2im(reset.data[0]))
        ])
        img_path = data['path']
        visualizer.save_images(webpage, visuals, img_path)
        webpage.save()

def video_encoding_predictions():
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'main.avi')
    opt.dataroot
    opt.isTrain = True
    opt.use_encoded_image = True

    model = TrafficSignNet.build()
    trained_model = train_neural_network(model)

    for i, data in enumerate(tqdm(dataset)):
        iter_start_time = time.time()
        total_steps += 1
        epoch_iter += 1

        # Forward pass
        losses, generated = model(
            Variable(data['label']),
            Variable(data['inst']),
            Variable(data['image']),
            Variable(data['feat']),
            infer=True
        )

        # Calculate and sum losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # Final loss calculation
        loss_D = (loss_dict['CNN'] + loss_dict['SVM']) * 0.5
        loss_G = loss_dict['LNT'] + loss_dict.get('GTRSB', 0) + loss_dict.get('main', 0)

        # Print errors
        errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
        time_per_batch = (time.time() - iter_start_time) / opt.batchSize
        visualizer.print_current_errors(epoch, epoch_iter, errors, time_per_batch)
        visualizer.plot_current_errors(errors, total_steps)

        # Display results
        visuals = OrderedDict([
            ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
            ('trained_image', util.tensor2im(generated.data[0])),
            ('real_image', util.tensor2im(data['image'][0]))
        ])
        visualizer.display_current_results(visuals, i, total_steps)

        # Save error
        np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.transform import resize

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1

NUM_TEST_IMAGES = 10
################################################# X_train, Y_train

dataset_dir = "C:/models/keras datasets/cracks"

original_images = os.path.join(dataset_dir, 'original')
masks = os.path.join(dataset_dir, 'msk_cracks')

images_id_list = os.listdir(original_images)
masks_id_lists = os.listdir(masks)

# len_train = int(len(images_id_list) * 0.98)
len_train = len(images_id_list)
X_train = np.zeros((len_train, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len_train, IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

X_test = np.zeros((len(images_id_list) - len_train, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(images_id_list) - len_train, IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

# Train dataset
for i, fname in enumerate(masks_id_lists):
    mask = imread(os.path.join(masks, fname), as_gray=True)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.expand_dims(mask, axis=-1)
    Y_train[i] = mask

for i, fname in enumerate(images_id_list):
    image = imread(os.path.join(original_images, fname), as_gray=True) * 255
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    image = np.expand_dims(image, axis=-1)
    X_train[i] = image

# Test dataset
# for i, fname in enumerate(images_id_list[len_train:]):
#     image = imread(os.path.join(original_images, fname), as_gray=True) * 255
#     image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     image = np.expand_dims(image, axis=-1)
#     X_test[i] = image
#
# for i, fname in enumerate(masks_id_lists[len_train:]):
#     mask = imread(os.path.join(masks, fname), as_gray=True)
#     mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     mask = np.expand_dims(mask, axis=-1)
#     Y_test[i] = mask

print('X_Train shape: {0}, Y_Train shape: {1}, X_Test shape: {2}, Y_Test shape: {3},'.
      format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

filepath = "cracks.h5"

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

callbacks_list = [earlystopper, checkpoint]

history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks_list)

# 'C:/models/keras datasets/CandPSegmentation/UpdatedSet/original/all/942028_RS_290_290RS070691_03045_RAW.jpg'

# test = imread('C:/models/keras datasets/CandPSegmentation/UpdatedSet/original/all/942028_RS_290_290RS070691_03045_RAW.jpg',
#                       as_gray=True) * 255
# import cv2
#
# annotated = cv2.imread('C:/models/keras datasets/CandPSegmentation/UpdatedSet/original/all/942028_RS_290_290RS070691_03045_RAW.jpg')
# test = resize(test, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
# test = np.expand_dims(test, axis=-1)
# test = np.expand_dims(test, axis=0)
#
# test_preds = model.predict(test)
#
# preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)
#
# kernel = np.ones((5, 5), 'uint8')
# img = cv2.resize(preds_test_thresh[0] * 255, (1024, 640))
# dilate_img = cv2.dilate(img, kernel, iterations=3)
# contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# coords = []
# color = (0, 0, 255)
#
# for c in contours:
#     coords.append(cv2.boundingRect(c))
# for c in coords:
#     cv2.rectangle(annotated, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), color, 2)
#
# cv2.imshow('0', annotated)
# cv2.waitKey()

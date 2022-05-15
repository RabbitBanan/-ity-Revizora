# from keras.models import Model, load_model
# from keras.layers import Input
# from keras.layers.core import Dropout, Lambda
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import layers, Input
# from keras import backend as K
# from
# import pandas as pd
# import numpy as np
# import os
# from sklearn.utils import class_weight
#
# import matplotlib.pyplot as plt
#
# from skimage.io import imread, imshow
# from skimage.transform import resize
#
# IMG_HEIGHT = 128
# IMG_WIDTH = 128
# IMG_CHANNELS = 1
#
# NUM_TEST_IMAGES = 10
# ################################################# X_train, Y_train
#
# dataset_dir = "C:/models/keras datasets/potholes"
#
# original_images = os.path.join(dataset_dir, 'original')
# masks = os.path.join(dataset_dir, 'msk_potholes')
#
# images_id_list = os.listdir(original_images)
# masks_id_lists = os.listdir(masks)
#
# # len_train = int(len(images_id_list) * 0.98)
# len_train = len(images_id_list)
# X_train = np.zeros((len_train, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_train = np.zeros((len_train, IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
#
# X_test = np.zeros((len(images_id_list) - len_train, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_test = np.zeros((len(images_id_list) - len_train, IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
#
# # Train dataset
# # for i, fname in enumerate(masks_id_lists):
# #     mask = imread(os.path.join(masks, fname), as_gray=True)
# #     mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
# #     mask = np.expand_dims(mask, axis=-1)
# #     Y_train[i] = mask
# #
# # for i, fname in enumerate(images_id_list):
# #     image = imread(os.path.join(original_images, fname), as_gray=True) * 255
# #     image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
# #     image = np.expand_dims(image, axis=-1)
# #     X_train[i] = image
#
# # Test dataset
# # for i, fname in enumerate(images_id_list[len_train:]):
# #     image = imread(os.path.join(original_images, fname), as_gray=True) * 255
# #     image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
# #     image = np.expand_dims(image, axis=-1)
# #     X_test[i] = image
# #
# # for i, fname in enumerate(masks_id_lists[len_train:]):
# #     mask = imread(os.path.join(masks, fname), as_gray=True)
# #     mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
# #     mask = np.expand_dims(mask, axis=-1)
# #     Y_test[i] = mask
#
# print('X_Train shape: {0}, Y_Train shape: {1}, X_Test shape: {2}, Y_Test shape: {3},'.
#       format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
#
# # Don't Show Warning Messages
# import warnings
# warnings.filterwarnings('ignore')
#
# import tensorflow as tf
#
# def get_custom_model():
#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     s = Lambda(lambda x: x / 255) (inputs)
#
#     c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
#     c1 = Dropout(0.1) (c1)
#     c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
#     p1 = MaxPooling2D((2, 2)) (c1)
#
#     c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
#     c2 = Dropout(0.1) (c2)
#     c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
#     p2 = MaxPooling2D((2, 2), padding='same') (c2)
#
#     c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
#     c3 = Dropout(0.2) (c3)
#     c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
#     p3 = MaxPooling2D((2, 2), padding='same') (c3)
#
#     c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
#     c4 = Dropout(0.2) (c4)
#     c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
#     p4 = MaxPooling2D((2, 2), padding='same')(c4)
#
#     c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
#     c5 = Dropout(0.3) (c5)
#     c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
#
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
#     c6 = Dropout(0.2) (c6)
#     c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
#
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
#     c7 = Dropout(0.2) (c7)
#     c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
#
#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
#     c8 = Dropout(0.1) (c8)
#     c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
#
#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
#     c9 = Dropout(0.1) (c9)
#     c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
#
#     outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
#     return Model(inputs=[inputs], outputs=[outputs])
#
# def unet(input_shape):
#     '''
#     Params: input_shape -- the shape of the images that are input to the model
#                            in the form (width_or_height, width_or_height,
#                            num_color_channels)
#     Returns: model -- a model that has been defined, but not yet compiled.
#                       The model is an implementation of the Unet paper
#                       (https://arxiv.org/pdf/1505.04597.pdf) and comes
#                       from this repo https://github.com/zhixuhao/unet. It has
#                       been modified to keep up with API changes in keras 2.
#     '''
#     inputs = Input(input_shape + (1,))
#
#     conv1 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
#
#     conv2 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
#
#     conv3 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D((2, 2), padding='same')(conv3)
#
#     conv4 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D((2, 2), padding='same')(drop4)
#
#     conv5 = Conv2D(filters=1024,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(filters=1024,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = layers.UpSampling2D(size=(2, 2))(drop5)
#     up6 = Conv2D(filters=512,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up6)
#     merge6 = layers.Concatenate(axis=3)([drop4, up6])
#     conv6 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv6)
#
#     up7 = layers.UpSampling2D(size=(2, 2))(conv6)
#     up7 = Conv2D(filters=256,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up7)
#     merge7 = layers.Concatenate(axis=3)([conv3, up7])
#     conv7 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv7)
#
#     up8 = layers.UpSampling2D(size=(2, 2))(conv7)
#     up8 = Conv2D(filters=128,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up8)
#     merge8 = layers.Concatenate(axis=3)([conv2, up8])
#     conv8 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv8)
#
#     up9 = layers.UpSampling2D(size=(2, 2))(conv8)
#     up9 = Conv2D(filters=64,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up9)
#     merge9 = layers.Concatenate(axis=3)([conv1, up9])
#     conv9 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(filters=2,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(conv9)
#
#     model = Model(inputs=inputs, outputs=conv10)
#
#     return model
#
# def dice_coef(y_true, y_pred):
#     '''
#     Params: y_true -- the labeled mask corresponding to an rgb image
#             y_pred -- the predicted mask of an rgb image
#     Returns: dice_coeff -- A metric that accounts for precision and recall
#                            on the scale from 0 - 1. The closer to 1, the
#                            better.
#     Citation (MIT License): https://github.com/jocicmarko/
#                             ultrasound-nerve-segmentation/blob/
#                             master/train.py
#     '''
#     y_true_f = tf.cast(K.flatten(y_true), dtype=tf.float64)
#     y_pred_f = tf.cast(K.flatten(y_pred), dtype=tf.float64)
#     intersection = K.sum(y_true_f * y_pred_f)
#     smooth = 1.0
#     return (2.0*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)
#
# filepath = "model.h5"
#
# model = unet((IMG_HEIGHT, IMG_WIDTH))
#
# model.compile(optimizer='adam', loss=dice_coef)
#
# earlystopper = EarlyStopping(patience=5, verbose=1)
#
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
#                              save_best_only=True, mode='min')
#
# callbacks_list = [earlystopper, checkpoint]
#
# # history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
# #                     callbacks=callbacks_list)
#
# x = imread("C:/Users/Ермаков И.В/Documents/917067_RS_290_290RS294095_18785_RAW.jpg", as_gray=True)
# x = resize(x, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
# x = np.expand_dims(x, axis=-1)
# res = model.predict(x)
# import cv2
# cv2.imshow('0', res * 255)

from keras_segmentation.models.unet import vgg_unet

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

model = vgg_unet(n_classes=1,  input_height=416, input_width=608)

model.train(
    train_images = "C:/models/keras datasets/potholes/original",
    train_annotations = "C:/models/keras datasets/potholes/msk_potholes",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="C:/models/keras datasets/potholes/original/883068_RS_290_290RS150131_16070_RAW.jpg",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
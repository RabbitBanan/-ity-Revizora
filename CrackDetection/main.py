import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# road semantic segmentation (mask extraction)

data_gen_args = dict(rotation_range=30,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     zoom_range=0.2,
                     shear_range=0.05,
                     validation_split=0.1)

image_datagen = ImageDataGenerator(**data_gen_args, rescale = 1.0/255)
mask_datagen = ImageDataGenerator(**data_gen_args, rescale = 1.0/255)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

original_dataset_dir = 'C:/models/keras datasets/CandPSegmentation/UpdatedSet'


image_generator = image_datagen.flow_from_directory(
    os.path.join(original_dataset_dir, 'original'),
    class_mode=None,
    color_mode='grayscale',
    target_size=(60, 60),
    seed=seed,
    subset='training')

val_image_generator = image_datagen.flow_from_directory(
    os.path.join(original_dataset_dir, 'original'),
    class_mode=None,
    color_mode='grayscale',
    target_size=(60, 60),
    seed=seed,
    subset='validation')

mask_generator = mask_datagen.flow_from_directory(
    os.path.join(original_dataset_dir, 'msk_potholes'),
    class_mode=None,
    color_mode='grayscale',
    target_size=(60, 60),
    seed=seed,
    subset='training')

val_mask_generator = mask_datagen.flow_from_directory(
    os.path.join(original_dataset_dir, 'msk_potholes'),
    class_mode=None,
    color_mode='grayscale',
    target_size=(60, 60),
    seed=seed,
    subset='validation')

train_generator = zip(image_generator, mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

from keras.layers import Conv2D, Dropout, MaxPooling2D, Input, Concatenate, UpSampling2D
from keras.models import Model, save_model, load_model

image_input = Input((60, 60, 1))

conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(image_input)
conv1 = Dropout(0.3)(conv1)
conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(pool1)
conv2 = Dropout(0.3)(conv2)
conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(pool2)
conv3 = Dropout(0.3)(conv3)
conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(conv3)

up1 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])
conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(up1)
conv4 = Dropout(0.3)(conv4)
conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv4)

up2 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])
conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(up2)
conv5 = Dropout(0.3)(conv5)
conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv5)

conv6 = Conv2D(1, 1, strides=1, activation='sigmoid', padding='same')(conv5)
outputs = conv6
model = Model(inputs=image_input, outputs=outputs)
model.summary()

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('segment_gray.h5', monitor='val_binary_accuracy', save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
# try:
#     history = model.fit_generator(
#             train_generator,
#             steps_per_epoch=100,
#             epochs=3,
#             validation_data=val_generator,
#             validation_steps=10,
#             callbacks=[checkpoint], workers=8)
# finally:
#     save_model(model, 'pothole_segmentation.h5')

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from IPython.display import display

model = load_model('pothole_segmentation.h5')

from keras.preprocessing import image
img = Image.open('C:/Signs/frame2.jpg')
img = img.convert('L')
original_size = img.size
original = np.asarray(img.copy())
img = img.resize((60, 60))
arr = np.asarray(img)

norm = arr.reshape((60, 60, 1))
norm = np.array([norm/255])

out = model.predict(norm)
out = out[0]
out = np.rint(out)
out = cv2.resize(out, original_size)
res = np.uint8(out * 255)
res = Image.fromarray(res, mode='L').filter(ImageFilter.ModeFilter(size=13))
res = np.asarray(res)

cv2.imshow('0', res)
cv2.waitKey()


masked = cv2.bitwise_and(original, original, mask = res)
cv2.imshow('0', masked)
cv2.waitKey()
# compare = np.concatenate((arr, out), axis=1)
# display(Image.fromarray(np.uint8(compare)))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def build_model(input_shape=(224,224,3), classes=6):
    inputs = keras.layers.Input(input_shape)
    x = keras.layers.Conv2D(32, (3, 3),activation='relu',padding='same')(inputs)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), )(x)

    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(classes, activation='relu')(x)
    return keras.Model(inputs, x)


batch_size=64
lr=1e-6
decay=0

datagen = keras.preprocessing.image.ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('imagenette_6class/train/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('imagenette_6class/validation/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)
# load and iterate test dataset
test_it = datagen.flow_from_directory('imagenette_6class/test/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)

model = build_model()

opt = keras.optimizers.Adam(lr=lr,decay=decay)


model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)

model.save('output/part1a.h5')
# loss = model.evaluate_generator(test_it, steps=24)

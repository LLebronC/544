import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import *
import seaborn as sn
import time
import json
# experiment='1_3'

def build_model(input_shape=(224,224,3), classes=6,dropout=True, batchnorm=True):
    inputs = keras.layers.Input(input_shape)
    x = keras.layers.Conv2D(32, (3, 3),activation='relu',padding='same')(inputs)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    if batchnorm:
     x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), )(x)

    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same')(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(classes, activation='relu')(x)
    return keras.Model(inputs, x)

experiment='1_5'
params=[True,True,True]#dropout,data augmentation,bacth norm
batch_size=32
lr=1e-10
decay=0
epochs=50

if params[1]:
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                rotation_range = 20,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                horizontal_flip = True)
    datagen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, )
else:
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, )
    datagen_test=datagen

# load and iterate training dataset
train_it = datagen.flow_from_directory('imagenette_6class/train/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('imagenette_6class/validation/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)
# load and iterate test dataset
test_it = datagen_test.flow_from_directory('imagenette_6class/test/',target_size=(224,224), class_mode='categorical', batch_size=batch_size,shuffle=False)
t1=time.time()
model = build_model(dropout=params[0], batchnorm=params[2])

opt = keras.optimizers.Adam(lr=lr,decay=decay)

callbacks=[
    keras.callbacks.History(),
    keras.callbacks.ModelCheckpoint('output/best_model_'+experiment+'.hdf5', monitor='val_accuracy', save_best_only=True),
keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10, restore_best_weights=True),
keras.callbacks.TerminateOnNaN()
    ]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#history = model.fit(train_it,epochs=epochs,validation_data=val_it, callbacks=callbacks)
t2=time.time()
took=t2-t1
model.load_weights('output/best_model_'+experiment+'.hdf5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('output/acc_'+experiment+'.png')

# Plot training & validation loss values

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('output/loss_'+experiment+'.png')


preds=np.argmax(model.predict(test_it),axis=1)
labels=test_it.labels
# some metrics
metrics = {
    'accuracy_score': accuracy_score(labels, preds),
    'balanced_accuracy_score': balanced_accuracy_score(labels, preds),
    'precision_recall_fscore_support': precision_recall_fscore_support(labels, preds,
                                                                       average='macro'),
    'time':took
}
print(metrics)
json.dump(metrics,open('output/results_'+experiment+'_real.json','w'))

plt.clf()
# computed and plot the confusion matrix
cm = confusion_matrix(labels, preds)
sns_plot = sn.heatmap(cm, xticklabels=test_it.class_indices.keys(), yticklabels=test_it.class_indices.keys(), cmap="YlGnBu")
plt.savefig('part_'+experiment+'_cm.png')
plt.clf()

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

def build_model(input_shape=(224,224,3),classes=4):
    model=keras.applications.resnet50.ResNet50(include_top=False,weights='imagenet')

    for layer in model.layers:
        if 'conv5_block3' not in layer.name:
            layer.trainable=False
        else:
            layer.trainable=True

    x = model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(classes, activation='sigmoid')(x)
    return keras.Model(model.input, x)

experiment='2_2'
batch_size=64
lr=1e-10
decay=0
epochs=200
datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)
datagen_test = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)

# load and iterate training dataset
train_it = datagen.flow_from_directory('food101_4class/train/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('food101_4class/validation/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)
# load and iterate test dataset
test_it = datagen_test.flow_from_directory('food101_4class/test/',target_size=(224,224), class_mode='categorical', batch_size=batch_size, shuffle=False)
t1=time.time()
model = build_model()

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
model.load_weights('output/best_model_2_1.hdf5')

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
    'time':took}
print(metrics)
json.dump(metrics,open('output/results_'+experiment+'.json','w'))
# computed and plot the confusion matrix
plt.clf()
cm = confusion_matrix(labels, preds)
sns_plot = sn.heatmap(cm, xticklabels=test_it.class_indices.keys(), yticklabels=test_it.class_indices.keys(), cmap="YlGnBu")
plt.savefig('part_'+experiment+'_cm.png')
plt.clf()

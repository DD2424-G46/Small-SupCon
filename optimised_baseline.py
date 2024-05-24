import sys
import os
import time
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.optimizers import SGD
import numpy as np

def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    # One-hot-encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # print("TRAIN NORM")
    # print(train_norm)
    # print("TEST NORM")
    # print(test_norm)
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # print("TRAIN NORM after division")
    # print(train_norm)
    # print("TEST NORM after division")
    # print(test_norm)

    return train_norm, test_norm

def normalise_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    mean = np.mean(train_norm, axis=(0,1,2), keepdims=True)
    std = np.std(train_norm, axis=(0,1,2), keepdims=True)

    train_norm = (train_norm - mean) / std
    test_norm = (test_norm - mean) / std

    return train_norm, test_norm

def define_model():

    # 1st VGG block
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # 2nd VGG block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # 3rd VGG block
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def summarize_diagnostics(history, num_epochs, batch_size, acc):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='Training set')
    plt.plot(history.history['val_loss'], color='orange', label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='Training set')
    plt.plot(history.history['val_accuracy'], color='orange', label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    filename = sys.argv[0].split('/')[-1]
    plt.savefig('results-baseline/epoch-test/' + filename + '_num_epochs='+ str(num_epochs) +'_batch_size='+ str(batch_size) + '_acc=' + str(acc) +'_plot.png')
    plt.close()

def run_test_harness():
    start_time = time.time()
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = normalise_pixels(trainX, testX)

    # trainX = trainX[:2000]
    # trainY = trainY[:2000]
    # testX = testX[:2000]
    # testY = testY[:2000]
    batch_size = 256
    num_epochs = 350
    model = define_model()
    # print(model.summary())
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.1)
    it_train = datagen.flow(trainX, trainY, batch_size=batch_size, subset='training')
    it_val = datagen.flow(trainX, trainY, batch_size=batch_size, subset='validation')
    history = model.fit(it_train, steps_per_epoch=45000 // batch_size, validation_data=it_val, validation_steps=5000 // batch_size, epochs=num_epochs)
    
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history, num_epochs, batch_size, acc)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)


run_test_harness()
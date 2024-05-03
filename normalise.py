import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import time
import numpy as np


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    # One-hot-encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def normalise_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    mean = np.mean(train_norm, axis=(0,1,2), keepdims=True)
    std = np.std(train_norm, axis=(0,1,2), keepdims=True)

    train_norm = (train_norm - mean) / std
    test_norm = (test_norm - mean) / std
    print("TRAIIINNN")
    print(train_norm)
    print("TESTS")
    print(test_norm)

    return train_norm, test_norm

def define_model():
    model = Sequential()
    # 1st VGG block
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    # 2nd VGG block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    # 3rd VGG block
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    	
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def summarize_diagnostics(history):
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='test')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')

    
    filename = sys.argv[0].split('/')[-1]
    filename = "HEJ"
    plt.savefig(filename + '_plot.png')
    plt.close()

def run_test_harness():
    start_time = time.time()
    
    trainX, trainY, testX, testY = load_dataset()
    trainX_subset = trainX[:20]
    trainY_subset = trainY[:20]
    testX_subset = testX[:20]
    testY_subset = testY[:20]

    trainX_subset, testX_subset = normalise_pixels(trainX_subset, testX_subset)
    # model = define_model()
    # history = model.fit(trainX_subset, trainY_subset, epochs=10, batch_size=64, validation_data=(testX_subset, testY_subset), verbose=0)
    # _, acc = model.evaluate(testX_subset, testY_subset, verbose=0)
    # print('> %.3f' % (acc * 100.0))
    # summarize_diagnostics(history)
    
    # # trainX, testX = prep_pixels(trainX, testX)
    # # model = define_model()
    # # history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
    # # _, acc = model.evaluate(testX, testY, verbose=0)
    # # print('> %.3f' % (acc * 100.0))
    # # summarize_diagnostics(history)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)

run_test_harness()
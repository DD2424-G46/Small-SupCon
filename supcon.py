import sys
from matplotlib import pyplot as plt
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Surpress some tensorflow messages 
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#BASELINE
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import time
from datetime import datetime

# Set the seed
tf.random.set_seed(42)
np.random.seed(42)
EXPERIMENTAL = False

def normalise_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    mean = np.mean(train_norm, axis=(0,1,2), keepdims=True)
    std = np.std(train_norm, axis=(0,1,2), keepdims=True)

    train_norm = (train_norm - mean) / std
    test_norm = (test_norm - mean) / std

    return train_norm, test_norm

# Keras implementation: https://keras.io/examples/vision/supervised-contrastive-learning/
# Used for ablation tests evaluating our model
def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    # augmented = data_augmentation(inputs)
    # outputs = resnet(augmented)
    outputs = resnet(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model

def create_CNN():
    model = Sequential()
    # 1st VGG block
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    # 2nd VGG block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    # 3rd VGG block
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))
    	
    model.add(Flatten())

    return model

# Based on the keras implementation: # Keras implementation: https://keras.io/examples/vision/supervised-contrastive-learning/
# Adapted to our baseline model
def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dense(projection_units, activation="relu", kernel_initializer='he_uniform')(features)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

    # Keras FC layer for the SupCon implementation 
    # for layer in encoder.layers:
    #     layer.trainable = trainable

    # inputs = keras.Input(shape=input_shape)
    # features = encoder(inputs)
    # features = layers.Dropout(dropout_rate)(features)
    # features = layers.Dense(hidden_units, activation="relu")(features)
    # features = layers.Dropout(dropout_rate)(features)
    # outputs = layers.Dense(num_classes, activation="softmax")(features)

    # model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate),
    #     loss=keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=[keras.metrics.SparseCategoricalAccuracy()],
    # )
    # return model


class SupConLoss(keras.losses.Loss): 
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature
        
    def __call__(self, labels, feature_vectors, sample_weight=None): 
         
        labels_actual = tf.squeeze(labels)
        one_hot_labels = tf.one_hot(labels_actual, num_classes) 
        one_hot_labels = tf.cast(one_hot_labels, tf.float32)
        pos_mask = tf.matmul(one_hot_labels, one_hot_labels, transpose_b=True)
        pos_mask = pos_mask - tf.linalg.diag(tf.linalg.diag_part(pos_mask))


        normalised_fv = tf.math.l2_normalize(feature_vectors, axis=1)
        normalised_fv = tf.cast(normalised_fv, tf.float32)

        similarity_matrix = tf.matmul(normalised_fv, normalised_fv, transpose_b=True)

        

        pos_similarity_matrix = tf.math.multiply(similarity_matrix, pos_mask)
    
        similarity_matrix = similarity_matrix - tf.linalg.diag(tf.linalg.diag_part(similarity_matrix))

        pos_sim_div_temp = tf.multiply(pos_similarity_matrix, 1/self.temperature)
        sim_div_temp = tf.multiply(similarity_matrix, 1/self.temperature)

        non_zero_temp_mask = pos_sim_div_temp != 0.0
        pos_sim_div_temp_pow_e = tf.where(non_zero_temp_mask, tf.math.exp(pos_sim_div_temp), pos_sim_div_temp)
        
        non_zero_temp_mask = sim_div_temp != 0.0
        sim_div_temp_pow_e = tf.where(non_zero_temp_mask, tf.math.exp(sim_div_temp), sim_div_temp)

        sim_div_temp_pow_e_sum = tf.reduce_sum(sim_div_temp_pow_e, 1)
        div_matrix = tf.math.divide(pos_sim_div_temp_pow_e, sim_div_temp_pow_e_sum)
   
        div_matrix = tf.where(div_matrix == 0, 1.0, div_matrix)
        log_div_matrix = tf.math.log(div_matrix)

        star = tf.reduce_sum(log_div_matrix, 1)

        heart = tf.reduce_sum(tf.multiply(pos_mask, -1), 1)
        
        non_zero_temp_mask = heart != 0.0 
        L_supp_out_vec = tf.where(non_zero_temp_mask, tf.math.divide(star, heart), heart)
        
        L_supp_out = tf.reduce_sum(L_supp_out_vec)

        return L_supp_out

        # Keras SupCon Implementation
        # # Normalize feature vectors
        # feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # # Compute logits
        # logits = tf.divide(
        #     tf.matmul(
        #         feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
        #     ),
        #     self.temperature,
        # )
        # return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

    


# Keras implementation: https://keras.io/examples/vision/supervised-contrastive-learning/
def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model


def summarize_diagnostics(history, loss_title, acc_title, accuracy):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(loss_title)
    plt.plot(history.history['loss'], color='blue', label='Training set')
    plt.plot(history.history['val_loss'], color='orange', label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if loss_title == "FC Cross-Entropy Loss":
        plt.subplot(1, 2, 2)
        plt.title(acc_title)
        plt.plot(history.history['sparse_categorical_accuracy'], color='blue', label='Training set')
        plt.plot(history.history['val_sparse_categorical_accuracy'], color='orange', label='Validation set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    filename = sys.argv[0].split('/')[-1]
    plt.savefig('results-new-dropout/results-tau/' + filename + '_' + loss_title + '_tau=' + str(temperature) + '_batch_size=' + str(batch_size) + '_acc=' + str(accuracy) + '_plot.png')
    plt.close()


def train_supcon(x_train, y_train, x_test, y_test):
    # Pre-training encoder
    encoder = create_CNN()
    # encoder = create_encoder()
    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupConLoss(temperature)
    )

    # Create a TensorBoard callback for debugging
    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
    #                                                 histogram_freq = 1,
    #                                                 profile_batch = '500,520')
    

    supcon_history = encoder_with_projection_head.fit(
       it_train, steps_per_epoch=45000 // batch_size, validation_data=it_val, validation_steps=5000 // batch_size, epochs=num_epochs_pretraining, #validation_data=(x_val, y_val)#, validation_split=0.1#, callbacks=[tboard_callback], #verbose=0
    )


    classifier = create_classifier(encoder, trainable=False)

    fully_connected_history = classifier.fit(it_train, steps_per_epoch=45000 // batch_size, validation_data=it_val, validation_steps=5000 // batch_size, epochs=num_epochs_training, #validation_data=(x_val, y_val)#, validation_split=0.1, #verbose=0
                             )

    accuracy = classifier.evaluate(x_test, y_test)[1]
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    summarize_diagnostics(supcon_history, "SupCon Loss", "SupCon Accuracy", accuracy)
    summarize_diagnostics(fully_connected_history,"FC Cross-Entropy Loss", "FC Cross-Entropy Accuracy", accuracy)

    

def experiment():
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

if __name__ == '__main__':
    # Tensorboard debugging
    # tf.debugging.experimental.enable_dump_debug_info(
    # "/tmp/tfdbg3_logdir",
    # tensor_debug_mode="FULL_HEALTH",
    # circular_buffer_size=-1)

    start_time = time.time()

    # Init of parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    
    learning_rate = 0.001
    batch_size = 256
    
    hidden_units = 512
    projection_units = 128
    num_epochs_training = 100
    num_epochs_pretraining = 200
    dropout_rate = 0.5
    temperature = 0.16

    # Load the train and test data splits
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Pre-processing
    x_train, x_test = normalise_pixels(x_train, x_test)

    # Data augmentation and training-validation-test split
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.1)
    it_train = datagen.flow(x_train, y_train, batch_size=batch_size, subset='training')
    it_val = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation')

    # Incompatible with Tensoflow metal for M2 Mac, hence ImageDataGenerator is used instead
    # data_augmentation = keras.Sequential( #CHANGE AUGMENTATION BASED ON STUDY?
    #     [
    #         layers.RandomFlip("horizontal"),
    #         layers.RandomRotation(0.02),
    #     ]
    # )

    if EXPERIMENTAL:
        experiment()
    else:
        train_supcon(x_train, y_train, x_test, y_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
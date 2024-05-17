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
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from datetime import datetime

# Set the seed
tf.random.set_seed(42)
np.random.seed(42)
EXPERIMENTAL = False

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

# def create_encoder():
#     resnet = keras.applications.ResNet50V2(
#         include_top=False, weights=None, input_shape=input_shape, pooling="avg"
#     )

#     inputs = keras.Input(shape=input_shape)
#     augmented = data_augmentation(inputs)
#     outputs = resnet(augmented)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
#     return model

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
    model.add(Dropout(0.2))
    # 3rd VGG block
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    	
    model.add(Flatten())

    return model

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model    


class SupConLoss(keras.losses.Loss): # supKån()
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature
        
    def __call__(self, labels, feature_vectors, sample_weight=None): # feature_vectors = z
         
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

    



def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

# def train_baseline(x_train, y_train, x_test, y_test):
#     encoder = create_encoder()

#     # encoder.summary()
#     classifier = create_classifier(encoder)
#     classifier.summary()


#     history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

#     accuracy = classifier.evaluate(x_test, y_test)[1]
#     print(f"Test accuracy: {round(accuracy * 100, 2)}%")

def summarize_diagnostics(history, loss_title, acc_title):
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
    plt.savefig(filename + '_' + loss_title + '_tau=' + str(temperature) + '_plot.png')
    plt.close()


def train_supcon(x_train, y_train, x_test, y_test):
    # Pre-training encoder
    encoder = create_CNN()

    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupConLoss(temperature)
    )

    #encoder_with_projection_head.summary()
    # Create a TensorBoard callback
    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
    #                                                 histogram_freq = 1,
    #                                                 profile_batch = '500,520')
    

    supcon_history = encoder_with_projection_head.fit(
        it_train, steps_per_epoch=steps, epochs=num_epochs, validation_data=(x_test, y_test)#, validation_split=0.1#, callbacks=[tboard_callback], #verbose=0
    )

    # Train classifier with frozen encoder
    classifier = create_classifier(encoder, trainable=False)

    fully_connected_history = classifier.fit(it_train, steps_per_epoch=steps, epochs=num_epochs, validation_data=(x_test, y_test)#, validation_split=0.1, #verbose=0
                             )

    # print(supcon_history.history.keys())
    # print(fully_connected_history.history.keys())
    summarize_diagnostics(supcon_history, "SupCon Loss", "SupCon Accuracy")
    summarize_diagnostics(fully_connected_history,"FC Cross-Entropy Loss", "FC Cross-Entropy Accuracy")

    accuracy = classifier.evaluate(x_test, y_test)[1]
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

def experiment():
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    # x = tf.constant([[1e-38, 1e-37, 0.000000000000000000000000000000000001]])
    # y = tf.constant([[10.0, 100.0, 1000.0]])
    # # x_sum = tf.reduce_sum(x, 1) # [3 3]
    # # y = tf.constant([[2, 0, 0], [0, 2, 2]])
    # # y_sum = tf.reduce_sum(y, 1) # [6 6]

    # mul = tf.math.log(x)
    # tf.print(mul)

if __name__ == '__main__': # REFER TO WEBSITE FOR INSPO
    # tf.debugging.experimental.enable_dump_debug_info(
    # "/tmp/tfdbg3_logdir",
    # tensor_debug_mode="FULL_HEALTH",
    # circular_buffer_size=-1)

    start_time = time.time()

    num_classes = 10
    input_shape = (32, 32, 3)

    # Minibatch params
    learning_rate = 0.001
    batch_size = 265
    # batch_size = 1000
    hidden_units = 512
    projection_units = 128
    num_epochs = 100
    dropout_rate = 0.5
    temperature = 0.05

     # Load the train and test data splits
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # SUBSET?
    # x_train = x_train[:5000]
    # y_train = y_train[:5000]
    # x_test = x_test[:5000]
    # y_test = y_test[:5000]
    # VALIDATION SET
    # x_val 


    # Display shapes of train and test datasets
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    # data_augmentation = keras.Sequential( #CHANGE AUGMENTATION BASED ON STUDY?
    #     [
    #         layers.Normalization(),
    #         layers.RandomFlip("horizontal"),
    #         layers.RandomRotation(0.02),
    #     ]
    # )


    # # Setting the state of the normalization layer.

    # data_augmentation.layers[0].adapt(x_train)
    x_train, x_test = prep_pixels(x_train, x_test)

    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    it_train = datagen.flow(x_train, y_train, batch_size=batch_size)
    steps = int(x_train.shape[0]/batch_size)

    if EXPERIMENTAL:
        experiment()
    else:
        train_supcon(x_train, y_train, x_test, y_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
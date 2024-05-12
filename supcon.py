import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow_addons as tfa
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
import time

# Set the seed
tf.random.set_seed(42)
np.random.seed(42)
EXPERIMENTAL = False

def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
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
    model.add(Dropout(0.2))
    # 3rd VGG block
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    	
    model.add(Flatten())
    # model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='softmax'))

    # opt = SGD(learning_rate=0.001, momentum=0.9)
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
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
        
        normalised_fv = tf.math.l2_normalize(feature_vectors, axis=1)
        # tf.print("first normalized element in normfv")   
        # tf.print(normalised_fv[0])
        labels_actual = tf.squeeze(labels)
        normalised_fv = tf.cast(normalised_fv, tf.float32)
        # 1. numsamples x numsamples
        similarity_matrix = tf.matmul(normalised_fv, normalised_fv, transpose_b=True)
        # tf.print("Frst sim")
        # tf.print(similarity_matrix[0][0])
        # 2. # numsamples x numclasses
        one_hot_labels = tf.one_hot(labels_actual, num_classes) 
        one_hot_labels = tf.cast(one_hot_labels, tf.float32)
        
        # 3. numsamples x numsamples
        pos_mask = tf.matmul(one_hot_labels, one_hot_labels, transpose_b=True)
        pos_mask = pos_mask - tf.linalg.diag(tf.linalg.diag_part(pos_mask))

        # 4. numsamples x numsamples
        pos_similarity_matrix = tf.math.multiply(similarity_matrix, pos_mask)
    
        # 5. numsamples x numsamples
        similarity_matrix = similarity_matrix - tf.linalg.diag(tf.linalg.diag_part(similarity_matrix))

        # check
        pos_sim_div_temp = tf.multiply(pos_similarity_matrix, 1/self.temperature)
        sim_div_temp = tf.multiply(similarity_matrix, 1/self.temperature)

        non_zero_temp_mask = pos_sim_div_temp != 0.0
        pos_sim_div_temp_pow_e = tf.where(non_zero_temp_mask, tf.math.exp(pos_sim_div_temp), pos_sim_div_temp)
        # tf.print("pos_sim_div_temp_pow_e")
        # tf.print(pos_sim_div_temp_pow_e)    
        
        non_zero_temp_mask = sim_div_temp != 0.0
        sim_div_temp_pow_e = tf.where(non_zero_temp_mask, tf.math.exp(sim_div_temp), sim_div_temp)
        # tf.print("sim_div_temp")
        # tf.print(sim_div_temp)
        # tf.print("sim_div_temp_pow_e")
        # tf.print(sim_div_temp_pow_e)

        sim_div_temp_pow_e_sum = tf.reduce_sum(sim_div_temp_pow_e, 1)
        div_matrix = tf.math.divide(pos_sim_div_temp_pow_e, sim_div_temp_pow_e_sum)
        # tf.print("div_matrix")
        # tf.print(div_matrix)

        # non_zero_temp_mask = div_matrix != 0.0 # ÄNDRA TILL POS MASK
        # log_div_matrix = tf.where(non_zero_temp_mask, tf.math.log(div_matrix), div_matrix) # NATYRAL???????
        # # tf.print("div_matrix")
        # # tf.print(div_matrix)
        # # tf.print("tf.clip_by_value(div_matrix, 1e-6, 1.)")
        # # tf.print(tf.clip_by_value(div_matrix, 1e-35, 1.))
        div_matrix = tf.where(div_matrix == 0, 1.0, div_matrix)
        log_div_matrix = tf.math.log(div_matrix)
        # tf.print("log_div_matrix")
        # tf.print(log_div_matrix)

        star = tf.reduce_sum(log_div_matrix, 1)
        # tf.print("star")
        # tf.print(star)


        heart = tf.reduce_sum(tf.multiply(pos_mask, -1), 1)
        # tf.print("heart")
        # tf.print(heart)
        
        non_zero_temp_mask = heart != 0.0 
        L_supp_out_vec = tf.where(non_zero_temp_mask, tf.math.divide(star, heart), heart)
        # tf.print("L_supp_out_vec")
        # tf.print(L_supp_out_vec)
        
        L_supp_out = tf.reduce_sum(L_supp_out_vec)
        # tf.print("L_supp_out")
        # tf.print(L_supp_out)

        return L_supp_out


        





        


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



def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

def train_baseline(x_train, y_train, x_test, y_test):
    encoder = create_encoder()

    # encoder.summary()
    classifier = create_classifier(encoder)
    classifier.summary()


    history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

    accuracy = classifier.evaluate(x_test, y_test)[1]
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

def train_supcon(x_train, y_train, x_test, y_test):
    # Pre-training encoder
    encoder = create_encoder()

    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupConLoss(temperature),
    )

    encoder_with_projection_head.summary()

    history = encoder_with_projection_head.fit(
        x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=0
    )

    # Train classifier with frozen encoder
    classifier = create_classifier(encoder, trainable=False)

    history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=0
                             )

    accuracy = classifier.evaluate(x_test, y_test)[1]
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

def main():
    encoder = create_encoder()
    supcoder = create_CNN()
    print("THEIR")
    print(encoder.summary())
    print("-"*20)
    print("OURS")
    print(supcoder.summary())
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
    hidden_units = 512
    projection_units = 128
    num_epochs = 2
    dropout_rate = 0.5
    temperature = 0.05

     # Load the train and test data splits
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # SUBSET?
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]



    # Display shapes of train and test datasets
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.02),
        ]
    )

    # Setting the state of the normalization layer.
    data_augmentation.layers[0].adapt(x_train)

    if EXPERIMENTAL:
        main()
    else:
        train_supcon(x_train, y_train, x_test, y_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
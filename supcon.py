import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Set the seed
tf.random.set_seed(42)
np.random.seed(42)

def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
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
        labels_actual = tf.squeeze(labels)
    
        # 1. numsamples x numsamples
        similarity_matrix = tf.matmul(normalised_fv, normalised_fv, transpose_b=True)
        # 2. # numsamples x numclasses
        one_hot_labels = tf.one_hot(labels_actual, num_classes) 
        one_hot_labels = tf.cast(one_hot_labels, tf.float32)
        
        # 3. numsamples x numsamples
        pos_mask = tf.matmul(one_hot_labels, one_hot_labels, transpose_b=True)
        
        # 4. numsamples x numsamples
        pos_similarity_matrix = tf.math.multiply(similarity_matrix, pos_mask)
    
        # 5. 
        similarity_matrix = similarity_matrix - tf.linalg.diag(tf.linalg.diag_part(similarity_matrix))
        pos_similarity_matrix = pos_similarity_matrix - tf.linalg.diag(tf.linalg.diag_part(pos_similarity_matrix))


        pos_sim_div_temp = tf.multiply(pos_similarity_matrix, 1/self.temperature)
        sim_div_temp = tf.multiply(similarity_matrix, 1/self.temperature)

        
        pos_mask = pos_mask - tf.linalg.diag(tf.linalg.diag_part(pos_mask))

        pos_sim_div_temp_pow_e = tf.math.exp(pos_sim_div_temp)
        pos_sim_div_temp_pow_e = tf.math.multiply(pos_sim_div_temp_pow_e, pos_mask)  # PERFORMANCE????
        sim_div_temp_pow_e = tf.math.exp(sim_div_temp)



        sim_div_temp_pow_e_inverse = tf.math.pow(sim_div_temp_pow_e, -1)

        div_matrix = tf.math.multiply(pos_sim_div_temp_pow_e, sim_div_temp_pow_e_inverse)



        non_zero_temp_mask = div_matrix != 0.0
        log_div_matrix = tf.where(non_zero_temp_mask, tf.math.log(div_matrix), div_matrix)

        # log_div_matrix = tf.math.log(div_matrix) # IS IT NATURAL?
        # log_div_matrix = tf.math.multiply(log_div_matrix, pos_mask)

        star = tf.reduce_sum(log_div_matrix, 1)


        pos_mask_sum = tf.reduce_sum(tf.multiply(pos_mask, -1), 1)


        heart = tf.math.pow(pos_mask_sum, -1)
        

        L_supp_out = tf.reduce_sum(tf.math.multiply(heart, star))
                
        
        
        # return L_supp_out


        





        


        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)



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
        x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs
    )

    # # Train classifier with frozen encoder
    # classifier = create_classifier(encoder, trainable=False)

    # history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

    # accuracy = classifier.evaluate(x_test, y_test)[1]
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")

def main():
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # x_sum = tf.reduce_sum(x, 1) # [3 3]

    # y = tf.constant([[2, 0, 0], [0, 2, 2]])
    # y_sum = tf.reduce_sum(y, 1) # [6 6]

    mul = tf.reduce_sum(-x,1)
    tf.print(mul)

if __name__ == '__main__': # REFER TO WEBSITE FOR INSPO
    # main()
    num_classes = 10
    input_shape = (32, 32, 3)

    # Minibatch params
    learning_rate = 0.001
    batch_size = 265
    hidden_units = 512
    projection_units = 128
    num_epochs = 1
    dropout_rate = 0.5
    temperature = 0.05

    # Load the train and test data splits
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # SUBSET?
    x_train = x_train[:5]
    y_train = y_train[:5]
    x_test = x_test[:5]
    y_test = y_test[:5]



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

    train_supcon(x_train, y_train, x_test, y_test)
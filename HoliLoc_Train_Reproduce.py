# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

seed_value = 29
tf.random.set_seed(41)
scaler = MinMaxScaler(feature_range=(0, 1))


def parse_args():
    parser = argparse.ArgumentParser(description='Holiloc Model Training Script')
    parser.add_argument('--train_data', type=str, help='Path to the training data CSV file', required=True)
    parser.add_argument('--img_feature_vectors', type=str, help='Path to the image feature vectors file', required=True)
    parser.add_argument('--sequence_embeddings', type=str, help='Path to the sequence embeddings file', required=True)
    parser.add_argument('--ppi_embeddings', type=str, help='Path to the PPI embeddings file', required=True)
    parser.add_argument('--output_model', type=str, help='Path to save the trained Holiloc model', required=True)
    return parser.parse_args()


def load_data(train_data_path, img_feature_vectors_path, sequence_embeddings_path, ppi_embeddings_path):
    train = pd.read_csv(train_data_path)
    y = np.array(train.drop(['Cluster_ID', 'UNIPROT', 'CELLLINE', 'IMAGE_URL'], axis=1))

    X_img = np.load(img_feature_vectors_path)

    X_seq = np.load(sequence_embeddings_path, allow_pickle=True)
    X_seq = np.vstack(X_seq)

    X_inta = np.load(ppi_embeddings_path)

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y, random_state=104, test_size=0.2, shuffle=True
    )
    X_train_seq_normalized = scaler.fit_transform(X_train_seq)
    X_test_seq_normalized = scaler.fit_transform(X_test_seq)
    X_train_seq = X_train_seq_normalized
    X_test_seq = X_test_seq_normalized

    X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
        X_inta, y, random_state=104, test_size=0.2, shuffle=True
    )
    X_train_int_normalized = scaler.fit_transform(X_train_int)
    X_test_int_normalized = scaler.fit_transform(X_test_int)
    X_train_int = X_train_int_normalized
    X_test_int = X_test_int_normalized

    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
        X_img, y, random_state=104, test_size=0.2, shuffle=True
    )

    return ( X_train_img, y_train_img, X_test_img, y_test_img, X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_train_int, y_train_int, X_test_int, y_test_int)


def create_image_model():

    image_model = Sequential()
    
    image_model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(224, 224, 3), name='imagemodel/conv2d_1'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), name='imagemodel/max_pooling2d_1'))
    image_model.add(Dropout(rate=0.3,seed=seed_value, name='imagemodel/dropout_1'))
    
    image_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', name='imagemodel/conv2d_2'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), name='imagemodel/max_pooling2d_2'))
    image_model.add(Dropout(rate=0.5,seed=seed_value, name='imagemodel/dropout_2'))
    
    image_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", name='imagemodel/conv2d_3'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), name='imagemodel/max_pooling2d_3'))
    image_model.add(Dropout(rate=0.3,seed=seed_value,name='imagemodel/dropout_3'))
    
    image_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', name='imagemodel/conv2d_4'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), name='imagemodel/max_pooling2d_4'))
    image_model.add(Dropout(rate=0.5,seed=seed_value, name='imagemodel/dropout_4'))
    
    image_model.add(Flatten(name='imagemodel/flatten'))
    
    image_model.add(Dense(128, activation='relu', name='imagemodel/dense_1'))
    image_model.add(Dropout(rate=0.3,seed=seed_value, name='imagemodel/dropout_5'))
    
    image_model.add(Dense(64, activation='relu', name='imagemodel/dense_2'))
    image_model.add(Dropout(rate=0.3,seed=seed_value, name='imagemodel/dropout_6'))
    
    image_model.add(Dense(22, activation='sigmoid', name='imagemodel/output_layer'))

    return image_model


def train_image_model(X_train_img, y_train_img, X_test_img, y_test_img):
    model_img = create_image_model()
    initial_learning_rate = 1e-4
    optimizer = Adam(learning_rate=initial_learning_rate)
    model_img.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-16)
    epochs = 30
    history_image = model_img.fit(X_train_img, y_train_img, validation_data=(X_test_img, y_test_img),
                                              epochs=epochs, callbacks=[reduce_lr])



def create_model_sequence():


    sequence_model = Sequential(name='sequence_model')
    sequence_model.add(Dense(units=256, input_shape=(1024,), name='sequence_model/dense_1'))
    sequence_model.add(BatchNormalization(name='sequence_model/batch_norm_1'))
    sequence_model.add(Activation('relu', name='sequence_model/activation_1'))
    sequence_model.add(Dropout(rate=0.1, seed=seed_value, name='sequence_model/dropout_1'))

    sequence_model.add(Dense(units=128, name='sequence_model/dense_2'))
    sequence_model.add(Activation('relu', name='sequence_model/activation_2'))
    sequence_model.add(Dropout(rate=0.2, seed=seed_value, name='sequence_model/dropout_2'))

    sequence_model.add(Dense(units=64, name='sequence_model/dense_3'))
    sequence_model.add(Activation('relu', name='sequence_model/activation_3'))
    sequence_model.add(Dropout(rate=0.1, seed=seed_value, name='sequence_model/dropout_3'))

    sequence_model.add(Dense(units=32, name='sequence_model/dense_4'))
    sequence_model.add(Activation('relu', name='sequence_model/activation_4'))
    sequence_model.add(Dropout(rate=0.3, seed=seed_value, name='sequence_model/dropout_4'))

    sequence_model.add(Dense(units=22, activation='sigmoid', name='sequence_model/output_layer'))

    return sequence_model


def train_sequence_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq):
    model_seq = create_model_sequence()
    initial_learning_rate = 1e-4
    optimizer = Adam(learning_rate=initial_learning_rate)
    model_seq.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-16)
    epochs = 20
    history_sequence = model_seq.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq),
                                                epochs=epochs, callbacks=[reduce_lr])
  


def create_interactome_model():
    
    interactome_model = Sequential(name='interactome_model')

    interactome_model.add(Dense(units=128, input_shape=(224,), name='interactome_model/dense_1'))
    interactome_model.add(BatchNormalization(name='interactome_model/batch_norm_1'))
    interactome_model.add(Activation('relu', name='interactome_model/activation_1'))
    interactome_model.add(Dropout(rate=0.4,seed=seed_value, name='interactome_model/dropout_1'))

    interactome_model.add(Dense(units=64, name='interactome_model/dense_2'))
    interactome_model.add(Activation('relu', name='interactome_model/activation_2'))
    interactome_model.add(Dropout(rate=0.5,seed=seed_value,name='interactome_model/dropout_2'))

    interactome_model.add(Dense(units=64, name='interactome_model/dense_3'))
    interactome_model.add(Activation('relu', name='interactome_model/activation_3'))
    interactome_model.add(Dropout(rate=0.1,seed=seed_value, name='interactome_model/dropout_3'))

    interactome_model.add(Dense(units=32, name='interactome_model/dense_4'))
    interactome_model.add(Activation('relu', name='interactome_model/activation_4'))
    interactome_model.add(Dropout(rate=0.1,seed=seed_value, name='interactome_model/dropout_4'))

    interactome_model.add(Dense(units=32, name='interactome_model/dense_5'))
    interactome_model.add(BatchNormalization(name='interactome_model/batch_norm_2'))
    interactome_model.add(Activation('relu', name='interactome_model/activation_5'))
    interactome_model.add(Dropout(rate=0.1,seed=seed_value, name='interactome_model/dropout_5'))

    interactome_model.add(Dense(units=22, activation='sigmoid', name='interactome_model/output'))

    return interactome_model


def train_interactome_model(X_train_int, y_train_int, X_test_int, y_test_int):
    model_inta = create_interactome_model()
    initial_learning_rate = 1e-5
    optimizer = Adam(learning_rate=initial_learning_rate)
    model_inta.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-16)
    epochs = 200
    history_interactome = model_inta.fit(X_train_int, y_train_int, validation_data=(X_test_int, y_test_int),
                                              epochs=epochs, callbacks=[reduce_lr])



def create_fusion_model(model_img, model_seq, model_inta):

    model_img=model_img
    model_seq=model_seq
    model_inta=model_inta
    
    image_input = model_img.input
    sequence_input = model_seq.input
    interactome_input = model_inta.input

    image_representation = model_img.get_layer('imagemodel/dense_2').output
    image_representation = Dense(128, activation='relu', name='image_dense')(image_representation)

    sequence_representation = model_seq.get_layer('sequence_model/dense_3').output
    sequence_representation = Dense(128, activation='relu', name='sequence_dense')(sequence_representation)

    interactome_representation = model_inta.get_layer('interactome_model/dense_2').output
    interactome_representation = Dense(128, activation='relu', name='interactome_dense')(interactome_representation)

    merged_representation = Concatenate()([image_representation, sequence_representation, interactome_representation])

    dense1_fusion = Dense(units=1024, activation='relu', name='dense1_fusion')(merged_representation)
    bn1_fusion = BatchNormalization(name='bn1_fusion')(dense1_fusion)
    dropout1_fusion = Dropout(0.2, seed=seed_value, name='dropout1_fusion')(bn1_fusion)

    dense2_fusion = Dense(units=1024, activation='relu', name='dense2_fusion')(dropout1_fusion)
    bn2_fusion = BatchNormalization(name='bn2_fusion')(dense2_fusion)
    dropout2_fusion = Dropout(0.1, seed=seed_value, name='dropout2_fusion')(bn2_fusion)

    dense3_fusion = Dense(units=1024, activation='relu', name='dense3_fusion')(dropout2_fusion)
    bn3_fusion = BatchNormalization(name='bn3_fusion')(dense3_fusion)
    dropout3_fusion = Dropout(0.1, seed=seed_value, name='dropout3_fusion')(bn3_fusion)

    dense4_fusion = Dense(units=448, activation='relu', name='dense4_fusion')(dropout3_fusion)
    bn4_fusion = BatchNormalization(name='bn4_fusion')(dense4_fusion)
    dropout4_fusion = Dropout(0.2, seed=seed_value, name='dropout4_fusion')(bn4_fusion)

    dense5_fusion = Dense(units=704, activation='relu', name='dense5_fusion')(dropout4_fusion)
    bn5_fusion = BatchNormalization(name='bn5_fusion')(dense5_fusion)
    dropout5_fusion = Dropout(0.2, seed=seed_value, name='dropout5_fusion')(bn5_fusion)

    fusion_output = Dense(units=22, activation='sigmoid', name='fusion_output1')(dropout5_fusion)

    fusion_model = Model(inputs=[image_input, sequence_input, interactome_input], outputs=fusion_output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    fusion_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return fusion_model


def train_fusion_model(fusion_model, X_train_img, X_train_seq, X_train_int, y_train_seq, X_test_img, X_test_seq,
                       X_test_int, y_test_seq):
    history_fused =fusion_model.fit([X_train_img, X_train_seq, X_train_int], y_train_seq, epochs=25, batch_size=32, validation_data=([X_test_img, X_test_seq, X_test_int], y_test_seq))

    # Save the entire fusion model
    fusion_model.save('holiloc_local.h5')


def main():
    args = parse_args()

    # Load data
    (
        X_train_img, y_train_img, X_test_img, y_test_img,
        X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        X_train_int, y_train_int, X_test_int, y_test_int
    ) = load_data(
        args.train_data, args.img_feature_vectors, args.sequence_embeddings, args.ppi_embeddings
    )

    # Train image model
    model_img = create_image_model()
    train_image_model(X_train_img, y_train_img, X_test_img, y_test_img)

    # Train sequence model
    model_seq = create_model_sequence()
    train_sequence_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq)

    # Train interactome model
    model_inta = create_interactome_model()
    train_interactome_model(X_train_int, y_train_int, X_test_int, y_test_int)

    # Create and train the fusion model
    fusion_model = create_fusion_model(model_img, model_seq, model_inta)
    train_fusion_model(fusion_model, X_train_img, X_train_seq, X_train_int, y_train_seq, X_test_img, X_test_seq,
                       X_test_int, y_test_seq)

    # Save the entire fusion model
    fusion_model.save(args.output_model)

if __name__ == "__main__":
    main()

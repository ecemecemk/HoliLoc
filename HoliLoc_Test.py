# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description='HoliLoc Model Evaluation Script')
    parser.add_argument('--model_path', type=str, help='Path to the pre-trained HoliLoc model', required=True)
    parser.add_argument('--test_data', type=str, help='Path to the test data CSV file', required=True)
    parser.add_argument('--img_feature_vectors', type=str, help='Path to the image feature vectors file', required=True)
    parser.add_argument('--sequence_embeddings', type=str, help='Path to the sequence embeddings file', required=True)
    parser.add_argument('--ppi_embeddings', type=str, help='Path to the PPI embeddings file', required=True)
    return parser.parse_args()

best_threshold=0.11560000000000001


def main():
    args = parse_args()

    # Load Model
    HoliLoc = load_model(args.model_path)

    # Load Test Data
    test = pd.read_csv(args.test_data)
    t = np.array(test.drop(['Cluster_ID', 'UNIPROT', 'CELLLINE', 'IMAGE_URL'], axis=1))

    # Load Image Feature Vectors
    T_img = np.load(args.img_feature_vectors)

    # Load Sequence Embeddings
    T_seq = np.load(args.sequence_embeddings, allow_pickle=True)
    T_seq = np.vstack(T_seq)
    normalized_seq_test_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(T_seq)
    T_seq = normalized_seq_test_data

    # Load PPI Embeddings
    T_inta = np.load(args.ppi_embeddings)
    normalized_inta_test_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(T_inta)
    T_inta = normalized_inta_test_data

    # Get Predictions
    pred_HoliLoc = HoliLoc.predict([T_img, T_seq, T_inta])

    # Make Predictions Based on Best Threshold
    outcome_holiloc = np.where(pred_HoliLoc < best_threshold, 0, 1)

    # Display Classification Report
    class_names = ['Actin_filaments', 'Aggresome', 'Cell_junctions', 'MTOC', 'Centrosome',
                   'Cytoplasmic_bodies', 'Cytosol', 'ER', 'Focal_adhesion_sites',
                   'Golgi_apparatus', 'Intermediate_filaments', 'Microtubules',
                   'Mitotic_spindle', 'Nuclear_bodies', 'Nuclear_membrane',
                   'Nuclear_speckles', 'Nucleoli', 'Nucleoli_fibrillar_center', 'PM',
                   'Nucleoplasm', 'Mitochondria', 'Cytokinetic_bridge']

    print(classification_report(t, outcome_holiloc, target_names=class_names))


if __name__ == "__main__":
    main()

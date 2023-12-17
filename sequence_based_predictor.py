import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import h5py

def predict_protein_location(target_protein, sequence_model_path, sequence_embeddings_path):
    best_threshold_sequence = 0.0369
    class_names = ["Actin_filaments", "Aggresome", "Cell_junctions", "MTOC", "Centrosome",
                   "Cytoplasmic_bodies", "Cytosol", "ER", "Focal_adhesion_sites",
                   "Golgi_apparatus", "Intermediate_filaments", "Microtubules",
                   "Mitotic_spindle", "Nuclear_bodies", "Nuclear_membrane",
                   "Nuclear_speckles", "Nucleoli", "Nucleoli_fibrillar_center",
                   "PM", "Nucleoplasm", "Mitochondria", "Cytokinetic_bridge"]

    # Load Sequence Model

    sequence_model = load_model(sequence_model_path)

    # Get Sequence Embedding

    ids = []
    embed = []
    target_embedding_sequence = None  # Initialize to None
    try:
        with h5py.File(sequence_embeddings_path, "r") as file:
            for sequence_id, embedding in file.items():
                ids.append(sequence_id)
                embed.append(np.array(embedding))
                if sequence_id == target_protein:
                    target_embedding_sequence = np.array(embedding)

        # Check if target_embedding_sequence is still None after the try block
        if target_embedding_sequence is None:
            raise IndexError  # Simulate the IndexError if target_embedding_sequence is not found

        sequence_embeddings_array_2d = target_embedding_sequence.reshape(-1, 1)  # to give normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        sequence_embeddings_array_normalized = scaler.fit_transform(sequence_embeddings_array_2d)
        sequence_embeddings_array_normalized = sequence_embeddings_array_normalized.reshape(-1,)

    except IndexError:
        print(f"Sequence embedding not found for {target_protein}. Unable to make predictions.")
        return []

    
    # Get Prediction


    sequence_embeddings_single = np.expand_dims(sequence_embeddings_array_normalized, axis=0)

    pred_sequence = sequence_model.predict(sequence_embeddings_single)

    outcome_sequence = np.where(pred_sequence < best_threshold_sequence, 0, 1)

    predicted_classes = [class_names[i] for i, value in enumerate(outcome_sequence[0]) if value == 1]

    return predicted_classes



def parse_args():
    parser = argparse.ArgumentParser(description='Protein Subcellular Location Prediction')
    parser.add_argument('--target_protein', type=str, help='Target protein name (e.g., "P68431")', required=True)
    parser.add_argument('--sequence_model_path', type=str, help='Path to the Sequence model file', required=True)
    parser.add_argument('--sequence_embeddings_path', type=str, help='Path to the sequence embeddings file', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = predict_protein_location(args.target_protein, args.sequence_model_path, args.sequence_embeddings_path)
    print("Predicted Subcellular Locations:", predictions)

if __name__ == "__main__":
    main()


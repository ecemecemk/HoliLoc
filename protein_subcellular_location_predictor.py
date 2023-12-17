import argparse
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import h5py

def predict_protein_location(target_protein, image_path, holiloc_model_path, sequence_embeddings_path, ppi_embeddings_path):
    best_threshold_holiloc = 0.11560000000000001
    class_names = ["Actin_filaments", "Aggresome", "Cell_junctions", "MTOC", "Centrosome",
                   "Cytoplasmic_bodies", "Cytosol", "ER", "Focal_adhesion_sites",
                   "Golgi_apparatus", "Intermediate_filaments", "Microtubules",
                   "Mitotic_spindle", "Nuclear_bodies", "Nuclear_membrane",
                   "Nuclear_speckles", "Nucleoli", "Nucleoli_fibrillar_center",
                   "PM", "Nucleoplasm", "Mitochondria", "Cytokinetic_bridge"]

    # Load HoliLoc Model

    holiloc_model = load_model(holiloc_model_path)
    human_interactome = pd.read_csv(ppi_embeddings_path)

    # Load Image Data and Obtain Image Feature Vector

    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    image_rgb = image_rgb.astype(np.float32)
    image_rgb = image_rgb / 255
    image_feature_vector = np.stack(image_rgb)

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

        
        # Get PPI Embedding
    try:
        human_interactome['PPI_Embedding'] = human_interactome['PPI_Embedding'].apply(lambda x: eval(x))
        target_embedding_graph = np.array(human_interactome[human_interactome['UNIPROT'] == target_protein]['PPI_Embedding'].values[0])
        target_embedding_graph = target_embedding_graph.astype(float)
        target_embedding_graph_2d = target_embedding_graph.reshape(-1, 1)  # to give normalization
        PPI_embeddings_array_normalized = scaler.fit_transform(target_embedding_graph_2d)
        PPI_embeddings_array_normalized = PPI_embeddings_array_normalized.reshape(-1,)

    except IndexError:
        print(f"PPI embedding not found for {target_protein}. Unable to make predictions.")
        return []
    
    
    # Get Prediction

    image_feature_vector_single = np.expand_dims(image_feature_vector, axis=0)
    sequence_embeddings_single = np.expand_dims(sequence_embeddings_array_normalized, axis=0)
    PPI_embeddings_single = np.expand_dims(PPI_embeddings_array_normalized, axis=0)

    pred_holiloc = holiloc_model.predict([image_feature_vector_single, sequence_embeddings_single, PPI_embeddings_single])

    outcome_holiloc = np.where(pred_holiloc < best_threshold_holiloc, 0, 1)

    predicted_classes = [class_names[i] for i, value in enumerate(outcome_holiloc[0]) if value == 1]

    return predicted_classes



def parse_args():
    parser = argparse.ArgumentParser(description='Protein Subcellular Location Prediction')
    parser.add_argument('--target_protein', type=str, help='Target protein name (e.g., "P68431")', required=True)
    parser.add_argument('--image_path', type=str, help='Path to the protein image', required=True)
    parser.add_argument('--holiloc_model_path', type=str, help='Path to the Holiloc model file', required=True)
    parser.add_argument('--sequence_embeddings_path', type=str, help='Path to the sequence embeddings file', required=True)
    parser.add_argument('--ppi_embeddings_path', type=str, help='Path to the PPI embeddings CSV file', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = predict_protein_location(args.target_protein, args.image_path, args.holiloc_model_path, args.sequence_embeddings_path, args.ppi_embeddings_path)
    print("Predicted Subcellular Locations:", predictions)

if __name__ == "__main__":
    main()


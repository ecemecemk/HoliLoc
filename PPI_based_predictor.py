import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predict_protein_location(target_protein, PPI_model_path, PPI_embeddings_path):
    best_threshold_PPI = 0.029400000000000003
    class_names = ["Actin_filaments", "Aggresome", "Cell_junctions", "MTOC", "Centrosome",
                   "Cytoplasmic_bodies", "Cytosol", "ER", "Focal_adhesion_sites",
                   "Golgi_apparatus", "Intermediate_filaments", "Microtubules",
                   "Mitotic_spindle", "Nuclear_bodies", "Nuclear_membrane",
                   "Nuclear_speckles", "Nucleoli", "Nucleoli_fibrillar_center",
                   "PM", "Nucleoplasm", "Mitochondria", "Cytokinetic_bridge"]

    # Load PPI Models

    PPI_model = load_model(PPI_model_path)
    human_interactome = pd.read_csv(PPI_embeddings_path)
        
        # Get PPI Embedding
    try:
        human_interactome['PPI_Embedding'] = human_interactome['PPI_Embedding'].apply(lambda x: eval(x))
        target_embedding_graph = np.array(human_interactome[human_interactome['UNIPROT'] == target_protein]['PPI_Embedding'].values[0])
        target_embedding_graph = target_embedding_graph.astype(float)
        target_embedding_graph_2d = target_embedding_graph.reshape(-1, 1)  # to give normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        PPI_embeddings_array_normalized = scaler.fit_transform(target_embedding_graph_2d)
        PPI_embeddings_array_normalized = PPI_embeddings_array_normalized.reshape(-1,)

    except IndexError:
        print(f"PPI embedding not found for {target_protein}. Unable to make predictions.")
        return []
    
    
    # Get Prediction

    PPI_embeddings_single = np.expand_dims(PPI_embeddings_array_normalized, axis=0)

    pred_PPI = PPI_model.predict(PPI_embeddings_single)

    outcome_PPI = np.where(pred_PPI < best_threshold_PPI, 0, 1)

    predicted_classes = [class_names[i] for i, value in enumerate(outcome_PPI[0]) if value == 1]

    return predicted_classes



def parse_args():
    parser = argparse.ArgumentParser(description='Protein Subcellular Location Prediction')
    parser.add_argument('--target_protein', type=str, help='Target protein name (e.g., "P68431")', required=True)
    parser.add_argument('--PPI_model_path', type=str, help='Path to the PPI model file', required=True)
    parser.add_argument('--PPI_embeddings_path', type=str, help='Path to the PPI embeddings CSV file', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = predict_protein_location(args.target_protein, args.PPI_model_path, args.PPI_embeddings_path)
    print("Predicted Subcellular Locations:", predictions)

if __name__ == "__main__":
    main()
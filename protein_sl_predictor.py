# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import h5py


        
class_names = ["Actin_filaments", "Aggresome", "Cell_junctions", "MTOC", "Centrosome",
               "Cytoplasmic_bodies", "Cytosol", "ER", "Focal_adhesion_sites",
               "Golgi_apparatus", "Intermediate_filaments", "Microtubules",
               "Mitotic_spindle", "Nuclear_bodies", "Nuclear_membrane",
               "Nuclear_speckles", "Nucleoli", "Nucleoli_fibrillar_center",
               "PM", "Nucleoplasm", "Mitochondria", "Cytokinetic_bridge"]

def predict_protein_location(model_type, **kwargs):
    if model_type == "image":
        return predict_protein_location_with_image(**kwargs)
    elif model_type == "sequence":
        return predict_protein_location_with_sequence(**kwargs)
    elif model_type == "PPI":
        return predict_protein_location_with_PPI(**kwargs)
    elif model_type == "HoliLoc":
        return predict_protein_location_with_HoliLoc(**kwargs)
    else:
        print(f"Invalid model type: {model_type}. Please choose from image, sequence, PPI, or HoliLoc.")
        return []


def prompt_for_image_args():
    return {
        'image_path': input('Enter the path to the protein image: '),
        'image_model_path': input('Enter the path to the Image Based Model File: ')
    }


def prompt_for_sequence_args():
    return {
        'target_protein': input('Enter the UniProt ID of the target protein: '),
        'sequence_model_path': input('Enter the path to the Sequence model file: '),
        'sequence_embeddings_path': input('Enter the path to the sequence embeddings file: ')
    }


def prompt_for_PPI_args():
    return {
        'target_protein': input('Enter the UniProt ID of the target protein: '),
        'PPI_model_path': input('Enter the path to the PPI model file: '),
        'PPI_embeddings_path': input('Enter the path to the PPI embeddings CSV file: ')
    }



def prompt_for_HoliLoc_args():
    return {
        'target_protein': input('Enter the UniProt ID of the target protein: '),
        'image_path': input('Enter the path to the protein image: '),
        'holiloc_model_path': input('Enter the path to the Holiloc model file: '),
        'sequence_embeddings_path': input('Enter the path to the sequence embeddings file: '),
        'ppi_embeddings_path': input('Enter the path to the PPI embeddings CSV file: ')
    }



def predict_protein_location_with_image(image_path, image_model_path):

    best_threshold_image = 0.09530000000000001

    # Load Model
    image_model = load_model(image_model_path)

    # Load Image and Get Image Feature Vector
    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    image_rgb = image_rgb.astype(np.float32)
    image_rgb = image_rgb / 255
    image_feature_vector = np.stack(image_rgb)

    # Get prediction
    image_feature_vector_single = np.expand_dims(image_feature_vector, axis=0)
    pred_image = image_model.predict(image_feature_vector_single)
    outcome_image = np.where(pred_image < best_threshold_image, 0, 1)
    predicted_classes_image = [class_names[i] for i, value in enumerate(outcome_image[0]) if value == 1]

    return predicted_classes_image


def predict_protein_location_with_sequence(target_protein, sequence_model_path, sequence_embeddings_path):

    best_threshold_sequence = 0.0369

    # Load Model
    sequence_model = load_model(sequence_model_path)

    # Load Sequence Embeddings and Normalize Target Embedding
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

    # Get prediction
    sequence_embeddings_single = np.expand_dims(sequence_embeddings_array_normalized, axis=0)
    pred_sequence = sequence_model.predict(sequence_embeddings_single)
    outcome_sequence = np.where(pred_sequence < best_threshold_sequence, 0, 1)
    predicted_classes_sequence = [class_names[i] for i, value in enumerate(outcome_sequence[0]) if value == 1]

    return predicted_classes_sequence


def predict_protein_location_with_PPI(target_protein, PPI_model_path, PPI_embeddings_path):
    
    best_threshold_PPI = 0.029400000000000003
    human_interactome = pd.read_csv(PPI_embeddings_path)
    # Load Model
    PPI_model = load_model(PPI_model_path)
        
    # Load PPI Embeddings and Normalize Target Embedding

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

    # Get prediction
    PPI_embeddings_single = np.expand_dims(PPI_embeddings_array_normalized, axis=0)
    pred_PPI = PPI_model.predict(PPI_embeddings_single)
    outcome_PPI = np.where(pred_PPI < best_threshold_PPI, 0, 1)
    predicted_classes_PPI = [class_names[i] for i, value in enumerate(outcome_PPI[0]) if value == 1]

    return predicted_classes_PPI


def predict_protein_location_with_HoliLoc(target_protein, image_path, holiloc_model_path, sequence_embeddings_path, ppi_embeddings_path):

    best_threshold_holiloc = 0.11560000000000001
    
    # Load Model
    holiloc_model = load_model(holiloc_model_path)

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
    with h5py.File(sequence_embeddings_path, "r") as file:
        for sequence_id, embedding in file.items():
            ids.append(sequence_id)
            embed.append(np.array(embedding))
            if sequence_id == target_protein:
                target_embedding_sequence = np.array(embedding)

    sequence_embeddings_array_2d = target_embedding_sequence.reshape(-1, 1)  # to give normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    sequence_embeddings_array_normalized = scaler.fit_transform(sequence_embeddings_array_2d)
    sequence_embeddings_array_normalized = sequence_embeddings_array_normalized.reshape(-1,)
    
    # Get PPI Embedding
    human_interactome = pd.read_csv(ppi_embeddings_path)
    human_interactome['PPI_Embedding'] = human_interactome['PPI_Embedding'].apply(lambda x: eval(x))
    target_embedding_graph = np.array(human_interactome[human_interactome['UNIPROT'] == target_protein]['PPI_Embedding'].values[0])
    target_embedding_graph = target_embedding_graph.astype(float)
    target_embedding_graph_2d = target_embedding_graph.reshape(-1, 1)  # to give normalization
    PPI_embeddings_array_normalized = scaler.fit_transform(target_embedding_graph_2d)
    PPI_embeddings_array_normalized = PPI_embeddings_array_normalized.reshape(-1,)

    # Get Prediction
    image_feature_vector_single = np.expand_dims(image_feature_vector, axis=0)
    sequence_embeddings_single = np.expand_dims(sequence_embeddings_array_normalized, axis=0)
    PPI_embeddings_single = np.expand_dims(PPI_embeddings_array_normalized, axis=0)

    pred_holiloc = holiloc_model.predict([image_feature_vector_single, sequence_embeddings_single, PPI_embeddings_single])

    outcome_holiloc = np.where(pred_holiloc < best_threshold_holiloc, 0, 1)

    predicted_classes_HoliLoc = [class_names[i] for i, value in enumerate(outcome_holiloc[0]) if value == 1]

    return predicted_classes_HoliLoc


def parse_args():
    parser = argparse.ArgumentParser(description='Protein Subcellular Location Prediction')
    parser.add_argument('--model_type', type=str, help='Type of the model (image, sequence, PPI, HoliLoc)', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_type == "image":
        additional_args = prompt_for_image_args()
    elif args.model_type == "sequence":
        additional_args = prompt_for_sequence_args()
    elif args.model_type == "PPI":
        additional_args = prompt_for_PPI_args()
    elif args.model_type == "HoliLoc":
        additional_args = prompt_for_HoliLoc_args()
    else:
        print(f"Invalid model type: {args.model_type}. Please choose from image, sequence, PPI, or HoliLoc.")
        return

    # Explicitly pass 'model_type' as a keyword argument
    all_args = {**vars(args), **additional_args, 'model_type': args.model_type}
    predictions = predict_protein_location(**all_args)
    print("Predicted Subcellular Locations:", predictions)



if __name__ == "__main__":
    main()
import argparse
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_protein_location(image_path, image_model_path):
    best_threshold_image = 0.09530000000000001

    class_names = ["Actin_filaments", "Aggresome", "Cell_junctions", "MTOC", "Centrosome",
                   "Cytoplasmic_bodies", "Cytosol", "ER", "Focal_adhesion_sites",
                   "Golgi_apparatus", "Intermediate_filaments", "Microtubules",
                   "Mitotic_spindle", "Nuclear_bodies", "Nuclear_membrane",
                   "Nuclear_speckles", "Nucleoli", "Nucleoli_fibrillar_center",
                   "PM", "Nucleoplasm", "Mitochondria", "Cytokinetic_bridge"]

    # Load Image Model

    ımage_model = load_model(ımage_model_path)

    # Load Image Data and Obtain Image Feature Vector

    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    image_rgb = image_rgb.astype(np.float32)
    image_rgb = image_rgb / 255
    image_feature_vector = np.stack(image_rgb)


    # Get Prediction

    image_feature_vector_single = np.expand_dims(image_feature_vector, axis=0)

    pred_image = image_model.predict(image_feature_vector_single)

    outcome_image = np.where(pred_image < best_threshold_image, 0, 1)

    predicted_classes = [class_names[i] for i, value in enumerate(outcome_image[0]) if value == 1]

    return predicted_classes



def parse_args():
    parser = argparse.ArgumentParser(description='Protein Subcellular Location Prediction')
    parser.add_argument('--image_path', type=str, help='Path to the protein image', required=True)
    parser.add_argument('--image_model_path', type=str, help='Path to the Image Based Model File', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = predict_protein_location( args.image_path, args.image_model_path)
    print("Predicted Subcellular Locations:", predictions)

if __name__ == "__main__":
    main()


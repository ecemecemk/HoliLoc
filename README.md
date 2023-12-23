# HoliLoc
Knowledge of subcellular localization (SL) of proteins is essential for drug development, systems biology, proteomics, and functional genomics. Due to the high costs associated with experimental studies, it has become crucial to develop computational systems to accurately predict proteins’ SLs. With different modes of biological data (e.g., sequences, biomedical images, unstructured text, etc.) becoming readily available to ordinary scientists, it is possible to leverage complementary types of data to increase both the performance and coverage of predictions. In this study, we propose HoliLoc, a new method for predicting protein SLs via multi-modal deep learning. Our approach makes use of three different types of data: 2D confocal microscopy images (Human Protein Atlas), amino acid sequences (UniProt), and protein-protein interactions - PPIs (IntAct) to predict SLs of proteins in a multi-label manner for 22 different cell compartments. HoliLoc takes protein data of 3 different modalities as input, encodes and embeds them (using ProtT5 language model for sequences and Node2Vec for PPIs), conducts learning separately on each data type (using a convolutional neural network - CNN for images and feed forward neural networks - FFN for the other two), and then concatenates these 3 embeddings and transform them to probabilities for 22 locations via another FFN. The system was trained in an end-to-end manner, and the performances were calculated on the unseen hold out test dataset which are significantly different from training samples (achieved by using UniRef50 clusters). The covered subcellular locations include Actin Filaments, Aggresome, Cell Junctions, Microtubule Organizing Center (MTOC), Centrosome, Cytoplasmic Bodies, Cytosol, Endoplasmic Reticulum (ER), Focal Adhesion Sites, Golgi Apparatus, Intermediate Filaments, Microtubules, Mitotic Spindle, Nuclear Bodies, Nuclear Membrane, Nuclear Speckles, Nucleoli, Nucleoli Fibrillar Center, Plasma Membrane (PM), Nucleoplasm, Mitochondria, and Cytokinetic Bridge. 

<img width="944" alt="HoliLoc_Schema" src="https://github.com/ecemecemk/HoliLoc/blob/main/Figures/HoliLoc_Schema.png">

# Model Structure

  HoliLoc enhances protein subcellular localization (SL) prediction through diverse data modalities: image, sequence, and interactome-protein-protein interaction (PPI). Our deep learning models, implemented in TensorFlow with Keras, ensure comprehensive insights into SL.
  
## Image Model

The Image Model employs a 20-layer CNN for image classification. It includes convolutional, pooling, dropout, flatten, and dense layers. Convolutional layers use filter sizes (16, 32, 64) with a (5, 5) kernel and ReLU activation. MaxPooling2D layers down-sample with a (2, 2) pool size. Dropout layers (0.3, 0.5 rates) aid regularization. Dense layers (128, 64 units) with ReLU activation follow, each with dropout layers (0.3) for abstraction recognition. The output layer (22 units) with sigmoid activation suits multi-label classification for protein location. Designed for (224, 224, 3) input data, the model is compiled with Adam optimizer and binary cross-entropy loss. For the detailed model structure: [HoliLoc Image Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/Figures/holiloc_image.svg).

## Sequence Model

The Sequence Model processes protein sequence embeddings through a FFN model with 16 layers. It includes dense, batch normalization, ReLU activation, and dropout layers, tailored for optimal classification performance. Batch normalization enhances stability, ReLU introduces non-linearity, and dropout combats overfitting. The output layer (22 units) with sigmoid activation is designed for sequence classification. The model, tailored for (1024,) input shape, is compiled using the Adam optimizer and binary cross-entropy loss.For the detailed model structure: [HoliLoc Sequence Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/Figures/holiloc_sequence.svg).


## Interactome Model (PPI)

PPI model composed of 20 layers. The model includes dense layers 
with varying units (128, 64, 64, 32, 32), batch normalization layers, activation layers 
with ReLU, and dropout layers with different dropout rates (0.4, 0.5, 0.1, 0.1, 0.1). 
Each of these layers collectively enables the model to understand complicated 
patterns within protein interactions. The final dense layer, comprising 22 units with a 
sigmoid activation, serves as the output layer for our classification task. 
The architecture is designed for input data with dimensions (224,). Model is 
compiled using the Adam optimizer with a binary cross-entropy loss. For the detailed model structure: [HoliLoc PPI Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/Figures/holiloc_PPI.svg).


## Model Fusion (HoliLoc)

HoliLoc Model leverages joint fusion, combining feature representations from intermediate layers of neural networks with data from three modalities—image, sequence, and interactome. This fusion creates a potent multi-modal neural network. The feature vector undergoes a FFN with 17 layers, including 6 dense layers, batch normalization, activation, and dropout layers. The output layer, using sigmoid activation, enables multi-label classification with 22 classes. Compiled with the Adam optimizer and binary cross-entropy loss, the model has a total of 4,663,606 parameters, with 4,654,390 trainable and an additional 9,216 non-trainable. For the detailed model structure: [HoliLoc Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/Figures/holiloc_fused.svg).

## Dependencies

Make sure you have the following dependencies installed before running the scripts:

- [pandas](https://pandas.pydata.org/): `pip install pandas` Version: 2.1.4
- [scikit-learn](https://scikit-learn.org/): `pip install scikit-learn` Version: 1.3.2
- [OpenCV](https://opencv.org/): `pip install opencv-python` Version: 4.8.1.78
- [TensorFlow](https://www.tensorflow.org/): `pip install tensorflow` Version: 2.15.0

## Image Dependencies
Protein confocal microscopy images need to be acquired in the following manner. The cells should be fixed in 4% formaldehyde and permeabilized with Triton X-100. The antibody for the target protein is combined with marker antibodies targeting gamma tubulin (to show microtubules) and calreticulin (to show the endoplasmic reticulum or ER), respectively. The nucleus is counterstained with 4',6-diamidino-2-phenylindole (DAPI). The primary antibodies are detected with the help of species-specific secondary antibodies labeled with different fluorophores (Alexa Fluor 488 for the protein of interest, Alexa Fluor 555 for microtubules, and Alexa Fluor 647 for ER). The cells are imaged using a laser scanning confocal microscope with a 63X objective. The different fluorophores are displayed as different channels in multicolor images, with the protein of interest shown in green, the nucleus in blue, microtubules in red, and the ER in yellow. You can use images in png or jpg format with any size.


## Pre-Trained Models
All HoliLoc and feature based( image,sequence,PPI) pre-trained model files can be obtained from here. [here](https://drive.google.com/file/d/17ugk4hviej1UFy2gKWChBP13A0Elwvk3/view?usp=drive_link).
Also, for each subcellular location HoliLoc and individual feature based models trained. Single location pre-trained models can be obtained [here](https://drive.google.com/file/d/1O99X19bUd82exS2aby_bpKDttS5qQnri/view?usp=drive_link).

-----------------------------------------------------------

# Predicting Protein Subcellular Location with HoliLoc Vesion: 0.1.0

* You can predict the subcellular location of your protein of interest by simply providing a confocal microscopy image and UniProt ID.
* You can use any model you like, HoliLoc or feature based models (image, sequence or PPI).
* Please download all necessary files from [here](https://drive.google.com/file/d/1PEnrMZsGI52zts6NF5EC-Nn5Nn2U0wNP/view?usp=drive_link). This file is consisting of embeddings, example image and multi-location models and protein_sl_predictor.py file. Unzip the file.
* Open terminal and navigate to the downloaded file's directory where protein_sl_predictor.py is located as well.
* Run the command below by specifying model type you want, image, sequence, PPI, or HoliLoc.

```
python protein_sl_predictor.py --model_type image, sequence, PPI, or HoliLoc

```

* The script will prompt you for the following information:
  
* Enter the UniProt ID of the target protein: e.g., P68431.
* Enter the path to the protein image: e.g., P68431.png.
* Enter the path to the Holiloc model file: HoliLoc.h5
* Enter the path to the sequence embeddings file: sequence_embeddings_all_proteins.h5
* Enter the path to the PPI embeddings CSV file: human_ppi_embeddings_all_proteins.csv


-------------------------------------------------------------------
# Training and Evaluating Models   

## Train

* If you want to reproduce HoliLoc model training please download necessery protein information, image feature vector, sequence and PPI embeddign files with HoliLoc_Train_Reproduce.py from [here](https://drive.google.com/file/d/13qtm6UMBX6KOUZ6XJ9_h5mMNl5LmRwk5/view?usp=drive_link). Unzip the file.
* Open terminal and navigate to the downloaded file's directory where  HoliLoc_Train_Reproduce.py is located as well.
* Run the command below.
* You will obtain the model file in the name you specifed in the "output_model" section as output.

```
python HoliLoc_Train_Reproduce.py --train_data HoliLoc_Train_Target.csv --img_feature_vectors Image_Feature_Vectors_Train.npy --sequence_embeddings Sequence_Embeddings_Train.npy --ppi_embeddings PPI_Embeddings_Train.npy --output_model holiloc_model.h5
```

* Provide the indicated arguments for your task.

* --train_data: Name of protein information file, HoliLoc_Train_Target.csv.
* --img_feature_vectors: Name of image feature vectors file, Image_Feature_Vectors_Train.npy.
* --sequence_embeddings_path: Name of sequence embeddings file, Sequence_Embeddings_Train.npy.
* --ppi_embeddings_path: Name of PPI embeddings file, PPI_Embeddings_Train.npy.
* --output_model: Name of the output HoliLoc model e.g. holiloc_model_repro.h5.

  

## Evaluate

* You can evaluate HoliLoc model with HoliLoc test dateset.
* Please download all necessary files from [here](https://drive.google.com/file/d/1qUM7t9D9RXMGu2BaFS1_u00aLUbXHOr8/view?usp=drive_link). Unzip the file.
* Open terminal and navigate to the downloaded file's directory where HoliLoc_Test.py is located as well.
* Add model file you want to evaluate to the directory, e.g. HoliLoc.h5.
* Run the command below.

```
python HoliLoc_Test.py --model_path HoliLoc.h5 --test_data HoliLoc_Test_Target.csv --img_feature_vectors Image_Feature_Vectors_Test.npy --sequence_embeddings Sequence_Embeddings_Test.npy --ppi_embeddings PPI_Embeddings_Test.npy
```

* Provide the indicated arguments for your task.

* --model_path: Path of model h5 file e.g. HoliLoc.h5.
* --test_data: Name HoliLoc_Test_Target.csv.
* --img_feature_vectors: Name of image feature vectors file, Image_Feature_Vectors_Test.npy.
* --sequence_embeddings: Name of sequence embeddings file, Sequence_Embeddings_Test.npy.
* --ppi_embeddings: Name of PPI embeddings file, PPI_Embeddings_Test.npy.



-------------------------------------------------------------------------------------
# License
Copyright (C) 2023 HUBioDataLab

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

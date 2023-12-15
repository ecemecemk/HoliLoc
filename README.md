# HoliLoc
Understanding protein subcellular locations is crucial in systems biology, drug development, and proteomics. Of the 20,394 human proteins, only 7,348 have experimentally verified localization annotations (UniProt, 2020_05). Leveraging AI and diverse data is essential. HoliLoc integrates amino acid sequences, interactome data, and protein-protein interactions for human proteins. Our deep learning model covers 22 locations in a multi-class, multi-label approach. Utilize HoliLoc for holistic protein subcellular localization insights.

Subcellular locations that HoliLoc covers are:
1. Actin Filaments
2. Aggresome
3. Cell Junctions
4. Microtubule Organizing Center (MTOC)
5. Centrosome
6. Cytoplasmic Bodies
7. Cytosol
8. Endoplasmic Reticulum (ER)
9. Focal Adhesion Sites
10. Golgi Apparatus
11. Intermediate Filaments
12. Microtubules
13. Mitotic Spindle
14. Nuclear Bodies
15. Nuclear Membrane
16. Nuclear Speckles
17. Nucleoli
18. Nucleoli Fibrillar Center
19. Plasma Membrane (PM)
20. Nucleoplasm
21. Mitochondria
22. Cytokinetic Bridge

<img width="944" alt="HoliLoc_Schema" src="https://github.com/ecemecemk/HoliLoc/assets/47942665/cb45cebb-acb6-433f-83fd-9a86c67627be">


## Classification Models 

  HoliLoc enhances protein subcellular localization (SL) prediction through diverse data modalities: image, sequence, and interactome-protein-protein interaction (PPI). Our deep learning models, implemented in TensorFlow with Keras, ensure comprehensive insights into SL.
  
### Image Model

The Image Model employs a 20-layer CNN for image classification. It includes convolutional, pooling, dropout, flatten, and dense layers. Convolutional layers use filter sizes (16, 32, 64) with a (5, 5) kernel and ReLU activation. MaxPooling2D layers down-sample with a (2, 2) pool size. Dropout layers (0.3, 0.5 rates) aid regularization. Dense layers (128, 64 units) with ReLU activation follow, each with dropout layers (0.3) for abstraction recognition. The output layer (22 units) with sigmoid activation suits multi-label classification for protein location. Designed for (224, 224, 3) input data, the model is compiled with Adam optimizer and binary cross-entropy loss. For the detailed model structure: [HoliLoc Image Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_image.svg).




### Sequence Model

The Sequence Model processes protein sequence embeddings through a FFN model with 16 layers. It includes dense, batch normalization, ReLU activation, and dropout layers, tailored for optimal classification performance. Batch normalization enhances stability, ReLU introduces non-linearity, and dropout combats overfitting. The output layer (22 units) with sigmoid activation is designed for sequence classification. The model, tailored for (1024,) input shape, is compiled using the Adam optimizer and binary cross-entropy loss.For the detailed model structure: [HoliLoc Sequence Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_sequence.svg).


### Interactome Model (PPI)

PPI model composed of 20 layers. The model includes dense layers 
with varying units (128, 64, 64, 32, 32), batch normalization layers, activation layers 
with ReLU, and dropout layers with different dropout rates (0.4, 0.5, 0.1, 0.1, 0.1). 
Each of these layers collectively enables the model to understand complicated 
patterns within protein interactions. The final dense layer, comprising 22 units with a 
sigmoid activation, serves as the output layer for our classification task (Figure 3.12). 
The architecture is designed for input data with dimensions (224,). Model is 
compiled using the Adam optimizer with a binary cross-entropy loss. For the detailed model structure: [HoliLoc PPI Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_PPI.svg).



### Model Fusion (HoliLoc)

HoliLoc Model leverages joint fusion, combining feature representations from intermediate layers of neural networks with data from three modalities—image, sequence, and interactome. This fusion creates a potent multi-modal neural network. The feature vector undergoes a FFN with 17 layers, including 6 dense layers, batch normalization, activation, and dropout layers. The output layer, using sigmoid activation, enables multi-label classification with 22 classes. Compiled with the Adam optimizer and binary cross-entropy loss, the model has a total of 4,663,606 parameters, with 4,654,390 trainable and an additional 9,216 non-trainable. For the detailed model structure: [HoliLoc Model Structure](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_fused.svg).


## HoliLoc Datasets
* Train dataset of the HoliLoc can be obtained here: [HoliLoc Train Dataset](https://drive.google.com/file/d/1GYRaLahUbSjXuyHJSdqpa043D5ZPRUXC/view?usp=drive_link)
* Test dataset of the HoliLoc can be obtained here: [HoliLoc Test Dataset](https://drive.google.com/file/d/1mvobd_R86PSKYEpcN4cCp-fvm91RcyYW/view?usp=drive_link)

Train and Test datasets have same structure composed of 26 columns. 22 of them are for one hot encoded location information (1: exist 0: does not exist) for subcellular location specified in column name. Other columns'descriptions are as follows:

* Cluster_ID --> UniRes Cluster ID of protein
* UNIPROT --> Uniprot ID of protein
* CELLLINE --> Cellline information of protein
* IMAGE_URL --> URL of Confocal fluorescence microscopy image showing all channels (green: Target protein, blue: Nucleus, red: Microtubules, yellow: ER)

## HoliLoc Feature Vectors and Embeddings

The resources to access the pre-generated feature vectors and embeddings from the training and test datasets of HoliLoc are as follows:

All of them supplied as zip file containing files for train and test separetely and in numpy array format.

* Image feature vectors resized to 224x224 pixels and normalized by dividing to 255: [Image Feature Vectors](https://drive.google.com/file/d/1cRPBHait35_IxuiNX9JEHwH_nt4ayv1r/view?usp=drive_link)
* Sequence embeddings produced with ProtT5, which are also found on train and test dataset sequence_embedding column provided as numpy array for the ease of user : [Sequence Embeddings](https://drive.google.com/file/d/17tiLTWf-_W8VumSn7W9T7u1Hawy5OZyP/view?usp=drive_link)
* PPI embedings obtained with graph learning and Node2Vec algorithm can be found here as numpy array for train and test separetely also: [PPI Embeddings](https://drive.google.com/file/d/15er4HxVvicOnAlr8IzRxxGdVEFURQ2bi/view?usp=drive_link)



-----------------------------------------------------------
# Predicting Protein Subcellular Location

Fine-tuned HoliLoc Model and necessary embedding files of sequence and PPI are available for download [here](https://drive.google.com/file/d/1K5oxBk3G-G5hIoTBEDT-gVoUtVRSqQEG/view?usp=drive_link). All you need to add here is png file of protein of interest.
## Dependencies

Make sure you have the following dependencies installed before running the script:

- [pandas](https://pandas.pydata.org/): `pip install pandas`
- [scikit-learn](https://scikit-learn.org/): `pip install scikit-learn`
- [OpenCV](https://opencv.org/): `pip install opencv-python`
- [TensorFlow](https://www.tensorflow.org/): `pip install tensorflow`


* To get the subcellular location prediction of protein of interest please open terminal and navigate to the directory where your script (protein_subcellular_location_predictor.py) is located and run the command below by changing the arguments according to your system.

```
python protein_location_predictor.py --target_protein P68431 --image_path path/to/your/image.png --holiloc_model_path path/to/your/HoliLoc_model.h5 --sequence_embeddings_path path/to/your/sequence_embeddings_all_proteins.h5 --ppi_embeddings_path path/to/your/human_ppı_embeddings_all_proteins.csv
```

* Change the indicated arguments for your task.

* --target_protein: UniProt ID of target protein.
* --image_path: Path of png file of protein confocal microscopy image.
* --holiloc_model_path: Path of HoliLoc model h5 file which is inside the downloaded file.
* --sequence_embeddings_path: Path of sequence_embeddings_all_proteins.h5 file which is inside the downloaded file.
* --ppi_embeddings_path: Path of human_ppı_embeddings_all_proteins.csv file which is inside the downloaded file.

-------------------------------------------------------------------
# Training and Evaluating Models   

## Train

* If you want to reproduce HoliLoc model training please download necessery protein information, image feature vector, sequence and PPI embeddign files from [here](https://drive.google.com/file/d/18_aUNxEcisyKatjuV-74zr4PXy7gh9bB/view?usp=drive_link).
  
* Open terminal and navigate to the directory where your script (HoliLoc_Train_Reproduce.py) is located and run the command below by changing the arguments according to your system.
  
```
python HoliLoc_Train_Reproduce.py --train_data path/to/HoliLoc_Train_Target.csv --img_feature_vectors path/to/Image_Feature_Vectors_Train.npy --sequence_embeddings path/to/Sequence_Embeddings_Train.npy --ppi_embeddings path/to/PPI_Embeddings_Train.npy --output_model holiloc_model.h5
```

* Change the indicated arguments for your task.


* --train_data: Path of protein information file which is inside the downloaded file with name HoliLoc_Train_Target.csv.
* --img_feature_vectors: Path of image feature vectors file which is inside the downloaded file with name Image_Feature_Vectors_Train.npy.
* --sequence_embeddings_path: Path of sequence embeddings file file which is inside the downloaded file with name Sequence_Embeddings_Train.npy.
* --ppi_embeddings_path: Path of sequence embeddings file file which is inside the downloaded file with name PPI_Embeddings_Train.npy.
* --output_model: Name of the output HoliLoc model e.g. holiloc_model_repro.h5.

## Evaluate

* If you want to evaluate HoliLoc model with HoliLoc test dateset necessary files are available for download [here](https://drive.google.com/file/d/1zcflN2qZNghK-gOrZ1EnhqesrSTtJ6tj/view?usp=drive_link).

* Open terminal and navigate to the directory where your script (HoliLoc_Test.py) is located and run the command below by changing the arguments according to your system.
```
python HoliLoc_Test.py --model_path path/to/model.h5 --test_data path/to/test_data.csv --img_feature_vectors path/to/image_feature_vectors.npy --sequence_embeddings path/to/sequence_embeddings.npy --ppi_embeddings path/to/ppi_embeddings.npy
```
* Change the indicated arguments for your task.

* --model_path: Path of HoliLoc model h5 file.
* --test_data: Path of protein information file which is inside the downloaded file with name HoliLoc_Test_Target.csv.
* --img_feature_vectors: Path of image feature vectors file which is inside the downloaded file with name Image_Feature_Vectors_Test.npy.
* --sequence_embeddings: Path of sequence embeddings file file which is inside the downloaded file with name Sequence_Embeddings_Test.npy.
* --ppi_embeddings: Path of sequence embeddings file file which is inside the downloaded file with name PPI_Embeddings_Test.npy.

-------------------------------------------------------------------------------------
# License
Copyright (C) 2023 HUBioDataLab

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

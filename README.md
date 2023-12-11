# HoliLoc
Understanding protein subcellular locations is crucial in systems biology, drug development, and proteomics. Of the 20,394 human proteins, only 7,348 have experimentally verified localization annotations (UniProt, 2020_05). Leveraging AI and diverse data is essential. HoliLoc integrates amino acid sequences, interactome data, and protein-protein interactions for human proteins. Our deep learning model covers 22 locations in a multi-class, multi-label approach. Utilize HoliLoc for holistic protein subcellular localization insights.


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

Train and Test datasets have same structure composed of 35 columns. Funcionality and meaning of each column as follows:

* Cluster_ID --> UniRes Cluster ID of protein
* UNIPROT --> Uniprot ID of protein
* GENE --> HGNC Symbol of associated gene of protein 
* GO_ID --> Cellular Component ID
* CELLLINE --> Cellline information of protein
* ORGANISIM --> Organism information of protein ( all is human)
* IMAGE_URL --> URL of Confocal fluorescence microscopy image showing all channels (green: Target protein, blue: Nucleus, red: Microtubules, ywllow: ER)
* IMAGE_URL_R --> URL of Confocal fluorescence microscopy image showing red channel only
* IMAGE_URL_G --> URL of Confocal fluorescence microscopy image showing green channel only
* IMAGE_URL_B --> URL of Confocal fluorescence microscopy image showing blue channel only
* IMAGE_URL_Y --> URL of Confocal fluorescence microscopy image showing yellow channel only
* B --> UniProt IDs of proteins interation with protein. (comma separeted if multiple)
* sequence_embedding --> 1024 sized amino-acid sequence embeddings obtained with ProtT5
* Actin_filaments --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Aggresome --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Cell_junctions --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* MTOC --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Centrosome --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Cytoplasmic_bodies --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Cytosol --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* ER --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Focal_adhesion_sites --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Golgi_apparatus --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Intermediate_filaments --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Microtubules --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Mitotic_spindle --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Nuclear_bodies --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Nuclear_membrane --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Nuclear_speckles --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Nucleoli --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Nucleoli_fibrillar_center --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* PM --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Nucleoplasm --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Mitochondria --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.
* Cytokinetic_bridge --> One hot encoded location information (1: exist 0: does not exist) for location name specified in column name.


## Predicting Protein Subcellular Localization Using Pre-trained Models

* Pre-trained HoliLoc models are available here: [HoliLoc Multi-Location Models](https://drive.google.com/file/d/13NdMsYFzJcg_I6E8n_AJKAVICjQ32d9l/view?usp=drive_link).
* If you want to get binary HoliLoc models for each subcellular localization HoliLoc models are available here: [HoliLoc Single-Location Models](https://drive.google.com/file/d/1O99X19bUd82exS2aby_bpKDttS5qQnri/view?usp=drive_link)



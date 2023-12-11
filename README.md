# HoliLoc

Understanding the subcellular locations of proteins is essential for 
investigations in systems biology, proteomics, drug development, and protein 
function. Out of the 20,394 reviewed human proteins, 7348 have localization 
annotation with experimental verification, according to UniProt's (version 2020_05). Additionally, there is very little data on suborganellar compartment localization. 
Approaches based on artificial intelligence have evolved nowadays, and data variety 
has grown. It has become more crucial to develop deep learning models that utilize 
data holistically. In addition, there is very little data on the localization of suborganellar compartments. The importance of utilizing different data resources 
holistically has increased in light of the development of artificial intelligence-based 
approaches and growing data diversity opportunities. It is only recently that the 
method of predicting subcellular localization by combining various forms of data 
from genome and proteome centered studies has started to draw interest. amino acid sequence, 
and interactome/ protein protein interaction (PPI) data are utilized together for 
human proteins and deep learning models are constructed by taking account the 
simplicity to mainly observe the data diversity impact. Hence, 3 different data types 
are obtained from the public biological data sources and organized according to the 
scope in which all proteins belong to human and containing only one member of the 
UniRef50 cluster to standardize and prevent redundancy. HoliLoc allows us to use different data together for protein subcellular localization for 22 different 
locations in multi-class and multi-label manner.


<img width="944" alt="HoliLoc_Schema" src="https://github.com/ecemecemk/HoliLoc/assets/47942665/cb45cebb-acb6-433f-83fd-9a86c67627be">


## Classification Models 

  HoliLoc leverages a diverse range of data modalities to enhance predictive 
accuracy and provide a comprehensive understanding of protein SL. Our approach 
outlined three key modalities, image, sequence and interactome-PPI. In this thesis, 
the deep learning models were constructed using the TensorFlow framework with the 
Keras.

### Image Model

  Image embeddings were employed as input for CNN, which has a total of 20 
layers, which includes convolutional layers, pooling layers, dropout layers, flatten 
layer, and dense layers. The convolutional layers which played an initial role in 
capturing fine details within the immunofluorescence images, utilize various filter 
sizes (16, 32, and 64) with a kernel size of (5, 5) and ReLU activation . 
MaxPooling2D layers follow each convolutional layer, employing a pool size of (2, 
2) for down-sampling and retaining critical image features, which reduces 
computational expense simultaneously enabling the model to recognize features in 
various regions of the image. Dropout layers with rates of 0.3 and 0.5, along with a 
specified seed value, are incorporated for regularization. The model flattens the 
output before progressing through dense layers, including two with 128 and 64 units, 
respectively, both employing ReLU activation. Dropout layers with rates of 0.3 are 
applied after each dense layer in which high-level abstractions and relationships 
between the detected features are recognized. The final dense layer, designated as the 
output layer, consists of 22 units with sigmoid activation, suitable for multi-label 
classification. The architecture is designed for image classification tasks on input 
data with dimensions (224, 224, 3). Sigmoid activation allows each output unit to 
independently produce values in the range [0, 1], which aligns well with multilabel 
classification where each class can be associated with multiple labels, in this case, it 
is appropriate for proteins’ presence in multiple locations. Model is compiled using 
the Adam optimizer with a binary cross-entropy loss. For the detailed model structure click [HoliLoc Image](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_image.svg).




### Sequence Model

  The protein sequence embeddings are given into the FFN model composed of 
16 layers in total, which include dense layers, batch normalization layers, activation 
layers (ReLU), and dropout layers. Each tailored to optimize its performance for the 
classification task. To enhance model stability, batch normalization is applied, 
followed by the ReLU activation function to introduce non-linearity. Additionally, 
dropout regularization is applied to cope with overfitting and to boost model 
robustness.(Figure 3.11). The output layer consists of 22 units with sigmoid 
activation, designed for sequence classification tasks. Model is designed for 
sequences with an input shape of (1024,). Model is compiled using the Adam 
optimizer with a binary cross-entropy loss. For the detailed model structure click [HoliLoc Sequence Diagram](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_sequence.svg).


### Interactome Model (PPI)

  Interactome FFN composed of 20 layers. The model includes dense layers 
with varying units (128, 64, 64, 32, 32), batch normalization layers, activation layers 
with ReLU, and dropout layers with different dropout rates (0.4, 0.5, 0.1, 0.1, 0.1). 
Each of these layers collectively enables the model to understand complicated 
patterns within protein interactions. The final dense layer, comprising 22 units with a 
sigmoid activation, serves as the output layer for our classification task (Figure 3.12). 
The architecture is designed for input data with dimensions (224,). Model is 
compiled using the Adam optimizer with a binary cross-entropy loss. For the detailed model structure click [HoliLoc PPI](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_PPI.svg).



### Model Fusion (HoliLoc)

  To harness the synergistic potential of three distinct modalities HoliLoc 
employs a technique known as joint fusion, also referred to as intermediate fusion. 
This process centers around the combination of feature representations learned from 
intermediate layers of neural networks with data from other modalities. The primary 
objective is to harmonize the variations in dimensionality and information content 
across these diverse modalities. The model incorporates three individual modules—
image, sequence, and interactome—that are fused to construct a powerful multimodal neural network. 
This feature vector is subsequently fed into a FFN, consisting 
of 17 layers in which 6 dense layers, batch normalization, activation, and dropout 
layers exist. The final output layer utilizes sigmoid activation for multi-label 
classification with 22 classes (Figure 3.13). Model is compiled using the Adam 
optimizer and binary cross-entropy loss. The model's architecture consists of a total 
of 4,663,606 parameters, with 4,654,390 being trainable and an additional 9,216 nontrainable. For the detailed model structure click [HoliLoc Fused](https://github.com/ecemecemk/HoliLoc/blob/main/holiloc_fused.svg).

## Predicting Protein Subcellular Localization Using Pre-trained Models

Pre-trained HoliLoc models are available here: [HoliLoc Multi-Location Models](https://drive.google.com/file/d/13NdMsYFzJcg_I6E8n_AJKAVICjQ32d9l/view?usp=drive_link).
If you want to get binary HoliLoc models for each subcellular localization HoliLoc models are available here: [HoliLoc Single-Location Models](https://drive.google.com/file/d/1O99X19bUd82exS2aby_bpKDttS5qQnri/view?usp=drive_link)



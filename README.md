# HoliLoc




## Classification Models 

  HoliLoc leverages a diverse range of data modalities to enhance predictive 
accuracy and provide a comprehensive understanding of protein SL. Our approach 
outlined three key modalities, image, sequence and interactome-PPI. In this thesis, 
the deep learning models were constructed using the TensorFlow framework with the 
Keras.

##Image Model

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
the Adam optimizer with a binary cross-entropy loss.

##Sequence Model

  The protein sequence embeddings are given into the FFN model composed of 
16 layers in total, which include dense layers, batch normalization layers, activation 
layers (ReLU), and dropout layers. Each tailored to optimize its performance for the 
classification task. To enhance model stability, batch normalization is applied, 
followed by the ReLU activation function to introduce non-linearity. Additionally, 
dropout regularization is applied to cope with overfitting and to boost model 
robustness.(Figure 3.11). The output layer consists of 22 units with sigmoid 
activation, designed for sequence classification tasks. Model is designed for 
sequences with an input shape of (1024,). Model is compiled using the Adam 
optimizer with a binary cross-entropy loss.

##Interactome Model (PPI)

  Interactome FFN composed of 20 layers. The model includes dense layers 
with varying units (128, 64, 64, 32, 32), batch normalization layers, activation layers 
with ReLU, and dropout layers with different dropout rates (0.4, 0.5, 0.1, 0.1, 0.1). 
Each of these layers collectively enables the model to understand complicated 
patterns within protein interactions. The final dense layer, comprising 22 units with a 
sigmoid activation, serves as the output layer for our classification task (Figure 3.12). 
The architecture is designed for input data with dimensions (224,). Model is 
compiled using the Adam optimizer with a binary cross-entropy loss.


##Model Fusion (HoliLoc)

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
of 4,663,606 parameters, with 4,654,390 being trainable and an additional 9,216 nontrainable.

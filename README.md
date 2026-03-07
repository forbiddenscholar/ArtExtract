# ArtExtract : Image classification using CNN and CRNN
This repo contains a notebook that has full implementation of the CNN and CRNN models for classifying the Artist, Genre and Style for the wikiart dataset.
## Approach :
Three architectures were implemented and compared : 
* CNN :
  * ResNet50
* CRNN :
  * ResNet50 + LSTM
  * ResNet50 + GRU
The notebook evaluates and compares the performance of these 3 architectures on the classification tasks.
## Evaluation Metrics :
- Accuracy
- Macro F1 score
- Classification report
- Class Distrtibution Analysis
- Confusion matrix
## Additional Details :
The notebook also explains :
- Design choices for each architecture
- Selected hyperparameters
- Handling data imbalance
- Challenges encountered and solutions
## Contents :
[ArtExtract_1.ipynb](https://github.com/forbiddenscholar/ArtExtract/blob/main/ArtExtract_1.ipynb) -> Complete implementation, experimentation and analysis

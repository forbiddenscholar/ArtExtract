# ArtExtract Evaluation Tasks
This repo contains the completed Task 1 and Task 2 for the artextract project test evaluation

---
## Taks 1 Painting classification
A model based on convolutional-recurrent architectures for classifying Style, Artist and Genre
### Approaches implemented : 
- ResNet50 (CNN baseline)
- ResNet50 + LSTM (CRNN)
- ResNet50 + GRU (CRNN)
### Evaluated using :
- Accuracy
- Macro F1-score
- Confusion matrix
- Class distribution analysis

---
## Task 2 Painiting similarity identification
A similarity finding system for a given query image (used National Gallery of Open Art Data)
### Approach :
- Feature extraction using ResNet50
- Similarity finding using
  * Cosine similarity
  * Euclidean Distance
### Evaluated using :
- Precision@k
- Recall@k
---

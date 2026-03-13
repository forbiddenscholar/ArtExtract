# ArtExtract : Painting Similarity using CNN Embeddings
This repo contains a notebook that has full implementation of the CNN model used for extracting the image embeddings and then identifying similar artworks such as paintings with similar composition, faces or poses.

Used National [Gallery of Art Open Data](https://github.com/NationalGalleryOfArt/opendata) dataset

## Approach :
A feature-based similarity pipeline was implemented using a pretrained CNN : 
* Feature Extraction :
  * ResNet50 pretrained on imagenet weights
* Similarity computation :
  * Cosine similarity
  * Euclidean distance
For a give query image, the system retrieves top-k similar images,
## Evaluation Metrics :
- Predision@k
- Recall@k
## Results :
Experiment showed that cosine similarity generally outperforms euclidean distance 
## Contents :
[ArtExtract_2.ipynb](https://github.com/forbiddenscholar/ArtExtract/blob/main/Task_2/ArtExtract_2.ipynb) -> Complete implementation, experimentation and analysis

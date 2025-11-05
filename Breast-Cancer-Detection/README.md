Breast cancer detection using machine learning:

Classifies breast tumors as Benign (B) or Malignant (M) using the WBDC dataset.  
Applies classical ML models to demonstrate how computational methods can aid medical diagnostics.

Overview:
Dataset: 30 cellular features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
Goal: Predict tumor type as Benign or Malignant
Features: radius, texture, smoothness, symmetry, etc.

Methods used:
Preprocessing: Label encoding, normalization - Clean & scale features 
PCA: Dimensionality reduction - Visualize separability 
SVM: Supervised classification, High accuracy - clear margin 
KNN:  Baseline classifier - Simplicity check 
K-Fold: CV, Cross-validation - Avoid overfitting 
KMeans: GMM, Unsupervised clustering - Explore natural grouping 

Insights
SVM performs best for binary separation.  
PCA visualization gives good interpretability.  
Unsupervised models roughly capture class structure.


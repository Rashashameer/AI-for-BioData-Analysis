Background:
Brain tumors are abnormal growths of cells in the brain that can be benign (non-cancerous) or malignant (cancerous).
Magnetic Resonance Imaging (MRI) is the most common non-invasive imaging technique used by radiologists to identify these tumors .
This project uses classical machine learning to classify brain MRI images as either containing a tumor ('yes') or no tumor ('no').

Overview:
The workflow involves preprocessing the images, applying a Support Vector Classifier (SVC) on the raw pixel data, and then applying Principal Component Analysis (PCA) to reduce dimensionality before training a second SVC to compare performance.

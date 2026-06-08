# Base Task: Fashion-MNIST Classifier

This directory holds the primary classification pipeline for the Fashion-MNIST dataset. 

## Overview
The `notebooks/` folder contains the standard data preprocessing, model definition, and training loops. The architecture was optimized to establish a solid baseline accuracy while aggressively preventing overfitting.

## Performance Metrics
* **Final Validation Accuracy:** ~91.6%
* **Final Validation Loss:** ~0.38

The `saved_models/` folder contains the `.pth` weights for the best-performing epoch. The metric plots and the final `submission.csv` predictions are exported to the root of this folder. 

As expected, the evaluation metrics show the primary failure mode is distinguishing between shirts, t-shirts, and pullovers due to their highly similar low-resolution pixel outlines.
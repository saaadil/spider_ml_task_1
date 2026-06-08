# Bonus Task: Autoencoder Dimensionality Analysis

This section explores the intrinsic dimensionality of the Fashion-MNIST dataset by forcing it through progressively smaller MLP autoencoder bottlenecks.

## The Experiment
I trained multiple iterations of the autoencoder, sweeping the latent dimension size from 4 up to 128, and plotted the resulting Mean Squared Error (MSE) on the test set. All visual outputs and plots are stored in the `results/` directory.

## Key Findings
* **Optimal Bottleneck:** 64 dimensions.
* **Observations:** The reconstruction loss drops sharply as the latent space increases to 32, and effectively flatlines at 64. Pushing the dimension to 128 yields almost zero return on investment for the reconstruction quality. This proves the core structural geometry of the dataset can be successfully compressed into a 64-dimensional vector space without meaningful data loss.
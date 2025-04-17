What's included:

Scripts for
* adding Gaussian noise
* applying a charge threshold
* applying input charge quantization
* shifting the center of a charge cluster

These codes currently modify the raw data from PixelAV to add noise and/or apply per-pixel charge-threshold and/or offset charge clusters. These modified files are then remade into the labels, recon2D, and recon3D parquet files. Functionality to submit as condor-job also added.
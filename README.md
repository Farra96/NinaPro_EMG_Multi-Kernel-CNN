# NinaPro_EMG_Multi-Kernel-CNN

This GitHub repository accounts for studying sEMG classification algorithms tailored for prosthetic hand applications. It provides:

- Various functions for extracting and formatting NinaPro databases.
- A fully parameterized Multi-Kernel model for easy tuning with grid-search techniques.
- Model variants to handle inhomogeneous data, including:
  - Triplet margin loss
  - Adversarial Network with Reversal Gradient
  - JS Divergence and Center Loss function.
- A more complex model that includes CBAM, featuring TCN, Feature Spatial Attention Module, and Feature Channel Attention Module.
- A framework for testing the effect of Transfer Learning on new users varying the number of data available.


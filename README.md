# NinaPro_EMG_Multi-Scale-CNN

This repository is based on the paper: A Multi-Scale CNN for Transfer Learning in sEMG-Based Hand Gesture Recognition for Prosthetic Devices
https://www.mdpi.com/1424-8220/24/22/7147

This GitHub repository accounts for studying sEMG classification algorithms tailored for prosthetic hand applications. It provides:

- Various functions for extracting and formatting NinaPro databases.
- A fully parameterized Multi-Kernel model for easy tuning with grid-search techniques.
- Model variants to handle inhomogeneous data, including:
  - Triplet margin loss
  - Adversarial Network with Reversal Gradient
  - JS Divergence and Center Loss function.
- A more complex model that includes CBAM, featuring TCN, Feature Spatial Attention Module, and Feature Channel Attention Module
- A framework for testing the effect of Transfer Learning on new users varying the number of data available.


The files Data_Processing_Utils, DATALOADERS, Training_type and MODELS contain the function used in the 2 following files.

The file DBX_Processing refers to how the data has been unpacked and structured in Pandas dataframe.

The file General_Models is the script used to run different training from prompt choosing configuration.

The file General_TL is the script used to run different Transfer Learning framework from prompt with custom configuration.








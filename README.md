# Cross Domain Plant Species Identification

**Members:** 

Alesandro Michael Ferdinand (101228984)

Bryan Austyn Ichsan (101229576)

Ng Jing Ping (101211418)


---


**Unit:** COS30082 - Applied Machine Learning


---


Drive Folders: https://drive.google.com/drive/folders/1TIup2Hk5IAztfO-q4QModHWNXcfdW7Cf?usp=share_link


## Prerequisites

- Open the Drive Folders and create a shortcut for it to the root path of your own Google Drive
- Ensure that your Jetson Nano has been installed with the required libraries (Tensorflow, etc.) and the other dependencies on Jetson Nano
- It is recommended to run the notebook files (`.ipynb`) using Google Colab.
- The codes for Jetson Nano will need to be copied into Jetson Nano to be run from the Jetson Nano itself.

## Manual

1. `AML_Models.ipynb` will be used to show the training process and the performance evaluation of the base CNN model, the CNN model with oversampled data, and the usage of MobileNetV2 as the transfer learning approach.
2. `AML_CycleGan.ipynb` will be used to show the training process and the performance evaluation of the proposed new approach to solve the imbalance data and classess issue using CycleGAN.
3. `AML_GUI.ipynb` will be used to run the GUI that can be used by the users to get the prediction of the species name of the plant image that the users fit into the GUI.
4. `AML_jetson.py` will be used to run inference (prediction) on Jetson Nano by using terminal.

# Feature Extraction Architecture Evaluation for Metastatic Tissue Classification

**Deep Learning Course Project**

## Overview
This project focuses on the binary classification of histopathology scans to detect metastatic vs. non-metastatic tissue. It evaluates multiple state-of-the-art convolutional neural network (CNN) architectures acting as feature extractors paired with a trainable multi-layer perceptron (MLP) classification head. This project utilizes the **PatchCamelyon (PCam)** dataset.

## Architectures Evaluated
The following backbone architectures (pre-trained on ImageNet) are evaluated by freezing their weights and training a custom classification head:
- **ResNet-50**
- **EfficientNet-B0**
- **VGG16**

## Dataset
The dataset used is the **PatchCamelyon (PCam)** dataset.
- A custom PyTorch `Dataset` is implemented to read images and labels lazily from HDF5 files (one sample at a time) to efficiently manage memory and avoid loading the entire dataset into RAM.
- **Images:** Processed from shape `(96, 96, 3)` uint8, cast to float32, normalized (divided by 255), and permuted to `(3, 96, 96)`.
- **Labels:** Squeezed from shape `(1, 1, 1)` uint8 to a scalar float32 (1 for metastatic, 0 for non-metastatic).

### Data Structure
To run the notebooks, ensure the dataset is placed within the `data/` directory. You will need the training, validation, and test HDF5 splits provided by the PCam dataset.

## Modeling Approach
1. **Backbone:** Pre-trained on ImageNet. All backbone parameters are frozen immediately after loading. The native classification head is bypassed using an identity mapping to directly expose the feature vectors.
2. **Head:** A small multi-layer perceptron (MLP) is trained from scratch. 
   - *Example (ResNet-50):* `Linear(2048, 512) → ReLU → Dropout(0.5) → Linear(512, 1) → Sigmoid`
   - *Example (EfficientNet-B0):* `Linear(1280, 512) → ReLU → Dropout(0.5) → Linear(512, 1) → Sigmoid`

## Repository Structure
- `ResNet-50_architecture.ipynb`: Implementation, training, cross-validation, and testing pipeline for ResNet-50.
- `efficientnet_b0.ipynb`: Implementation, training, cross-validation, and testing pipeline for EfficientNet-B0.
- `vgg16.ipynb`: Implementation, training, cross-validation, and testing pipeline for VGG16.
- `results.ipynb`: Summary and visualizations of the classification results and performance comparisons across the models.
- `data/`: Placeholder directory for storing PCam evaluation and training data (.h5 files).

## Running the Project
1. Configure your environment by installing required Python packages (PyTorch, Torchvision, h5py, scikit-learn, Jupyter, etc.).
2. Download the PatchCamelyon (PCam) dataset and extract the HDF5 files into the local `data/` folder.
3. Launch Jupyter Notebook (`jupyter notebook` or `jupyter lab`).
4. Execute the specific model notebooks (`ResNet-50_architecture.ipynb`, `efficientnet_b0.ipynb`, `vgg16.ipynb`). The flow for each notebook is:
   - Configure File Paths and Hyperparameters.
   - Run the training loop (which validates each epoch and saves a checkpoint of the best model).
   - Evaluate the newly trained model against the holdout test set.
5. Review the `results.ipynb` notebook for combined metrics profiling.

## Evaluation Metrics
Trained models are evaluated on the test set using two chief metrics:
- **Test Accuracy** (sigmoid activation thresholded at 0.5)
- **Test AUC-ROC** (utilizing raw sigmoid probabilities)

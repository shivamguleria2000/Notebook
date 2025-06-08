# Pneumonia Classification from Chest X-Rays

This project implements a deep learning model to classify chest X-ray images as either "Normal" or "Pneumonia". It uses a pre-trained ResNet-50 model on the PneumoniaMNIST dataset, a collection of medical images.

## Project Overview

The core of this project is a Python script that:
1.  Downloads and preprocesses the PneumoniaMNIST dataset.
2.  Handles class imbalance in the training data using a weighted random sampler.
3.  Implements a transfer learning approach by fine-tuning the final layer of a ResNet-50 model pre-trained on ImageNet.
4.  Trains the model and evaluates its performance using various metrics, including accuracy, precision, recall, F1-score, and AUC-ROC.
5.  Saves a confusion matrix plot (`confusion_matrix.png`) from the test set evaluation.

## Getting Started

### Prerequisites

You need Python 3 and the libraries listed in `requirements.txt`.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/shivamguleria2000/Notebook
    cd Notebook
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Training Script

To train the model and run the evaluation, simply execute the following command:

```bash
python pneumonia_classification_training_script.py
```

Alternatively, you can run the project using the Jupyter Notebook:

1.  Ensure you have Jupyter Notebook or JupyterLab installed:
    ```bash
    pip install notebook
    ```
2.  Start the Jupyter server:
    ```bash
    jupyter notebook
    ```
3.  Open the `Pneumonia_Classification_Training_Script.ipynb` file in the Jupyter interface and run the cells sequentially.

The script will perform the following actions:
- Download the dataset from MedMNIST.
- Train the model, showing validation metrics for each epoch.
- Evaluate the best model on the test set.
- Print a summary of performance metrics.
- Save a file named `confusion_matrix.png` with a plot of the results.

## Evaluation

The model's performance is evaluated using several key metrics suitable for medical diagnostic tasks:
-   **Accuracy**: Overall correct predictions.
-   **Precision, Recall, F1-Score**: To balance the trade-offs between false positives and false negatives, which is critical in medical contexts.
-   **AUC-ROC**: To measure the model's ability to distinguish between the two classes.

The final evaluation results on the test set are printed to the console at the end of the script execution. 
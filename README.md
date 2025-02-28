# Handwritten Digit Classification <a name="title"></a>

Supervised Learning of an RBF network to conduct Handwritten Digits Classification of MNIST dataset.

## Overview

This repository contains a Jupyter Notebook that will train a RBF network on a real image dataset of handwritten number digits. The notebook includes the following:

- **Robust Dataset**:
    - 60,000 images in training set, 10,000 images in test set
    - Each image is 28x28 pixels
    - The images are grayscale (no RGB colors)
    - Each image (data sample) contains one of 10 numeric digit $0, 1, 2, \ldots, 8, 9$
 
- **RBF network implementation** without using an outside library
- **90+% Prediction Accuracy** achieved on test set

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Attribution](#attribution)

## Getting Started

### Prerequisites

- **Python 3.8+** (or your preferred version)
- **Jupyter Notebook** (or JupyterLab)
- A list of Python libraries used in the notebook, for example:
  - `pandas`
  - `numpy`
  - `matplotlib`

### Installation

1. **Clone** the repository to your local machine:
   ```bash
   git clone https://github.com/henrylanday/Handwritten-Digit-Classification.git

2. **Navigate** into the repository:
   ```bash
   cd Handwritten-Digit-Classification

3. **Install** required Python packages:
   ```bash
   pip install -r requirements.txt


### Usage

1. **Open** the Jupyter Notebook (rbf_mnist.ipynb)
   ```bash
   jupyter notebook
2. **Navigate** to the notebook in the Jupyter interface and open it.
3. **Run the notebook** cells in order to reproduce the classifier


### Project Structure
   ~~~sh
your-repo-name/
    ├── pycache/
    ├── classifier.py
    ├── data/
    │   ├── mnist_test_data.npy
    │   ├── mnist_test_labels.npy
    │   ├── mnist_train_data.npy
    │   ├── mnist_train_labels.npy
    │   ├── rbf_dev_test.csv
    │   ├── rbf_dev_train.csv
    │   └── Screenshot 20... (example screenshot file)
    ├── kmeans.py
    ├── rbf_mnist.ipynb
    ├── rbf_net.py
    ├── README.md
    └── requirements.txt
  ~~~


- **__pycache__/**: Auto-generated cache files for Python modules.
- **classifier.py**: Python module containing classification logic or classes/functions for classification tasks.
- **data/**: Contains datasets and supporting files:
  - **mnist_test_data.npy**, **mnist_test_labels.npy**: NumPy arrays for MNIST test samples and labels.
  - **mnist_train_data.npy**, **mnist_train_labels.npy**: NumPy arrays for MNIST training samples and labels.
  - **rbf_dev_test.csv**, **rbf_dev_train.csv**: Example CSV files used for development and testing.
  - **Screenshot 20...**: A screenshot file.
- **kmeans.py**: Implements the K-means clustering algorithm.
- **rbf_mnist.ipynb**: Main Jupyter Notebook performing analyses/experiments (e.g., training the RBF network on MNIST).
- **rbf_net.py**: Implements the RBF (Radial Basis Function) network.
- **README.md**: Project documentation file (this file).
- **requirements.txt**: A list of Python dependencies needed to run the project.


### Results:
<img width="780" alt="Screenshot 2025-02-26 at 4 52 53 PM" src="https://github.com/user-attachments/assets/a9a932bd-5444-4827-b191-2b0f2d1a4663" />


## Created by Henry Landay <a name="attribution"></a>


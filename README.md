# Locally Normalized Soft Contrastive Clustering for Compact Clusters

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1%2B-orange.svg)

This repository contains the implementation of the paper:

**"Locally Normalized Soft Contrastive Clustering for Compact Clusters"**

[![LNSCC Train (CIFAR10)]](github/lnscc_train.gif)

[![LNSCC Test (CIFAR10)]](github/lnscc_test.gif)

## Overview

Clustering is a fundamental task in machine learning, aiming to group similar data points together. Traditional clustering methods often struggle with high-dimensional data and complex cluster structures. In this work, we propose **Locally Normalized Soft Contrastive Clustering (LNSCC)**, a novel approach that leverages contrastive learning and local normalization to produce compact and well-separated clusters.

This repository provides the implementation of LNSCC, including data preprocessing, feature extraction, and the training pipeline for clustering. The implementation is demonstrated on the CIFAR-10 dataset but can be adapted to other datasets with minimal modifications.

## Architecture

The project consists of three main scripts:

1. **`data_processing.py`**: Handles data loading, feature extraction, and construction of the weighted KNN graph.
2. **`lnscc_train.py`**: Implements the neural network model, training loop with losses, and visualization of clustering results.
3. **`main.py`**: Orchestrates the execution by sequentially running `data_processing.py` followed by `lnscc_train.py`.

## Installation

### Prerequisites

- **Python 3.8+**
- **CUDA** (for GPU acceleration)
- **Git**

### Clone the Repository

```bash
git clone https://github.com/xma24/lnscc.git
cd lnscc-clustering
```

### Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
# Activate the virtual environment
source venv/bin/activate
```

### Install Dependencies

Install the required Python packages using **`pip`**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: Ensure that you have a compatible version of PyTorch installed. You can customize the PyTorch installation based on your CUDA version by visiting PyTorch's official website.

### Running the Entire Pipeline

To execute both data preprocessing and model training sequentially, use the **`main.py`** script:

```bash
python main.py
```

### Results

After running the pipeline, the following outputs will be generated:

- **Pickle Files**: Preprocessed data, embeddings, and graph structures saved in the **`./data/`** directory.
- **Visualizations**: Scatter plots of embeddings with cluster assignments saved in the **`./work_dirs/`** directory, along with NMI and ARI scores.

### Citing This Work

If you use this implementation in your research, please cite our paper:

```bibtex
@inproceedings{ma2022locally,
  title={Locally normalized soft contrastive clustering for compact clusters},
  author={Ma, Xin and Kim, Won Hwa},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2022}
}
```

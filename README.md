# 🌟 CIFAR-10 Image Classification with Custom Mini-InceptionNet

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📌 Project Overview
This repository contains a complete deep learning pipeline for image classification on the CIFAR-10 dataset. Instead of relying on pre-trained models, I designed and trained a **custom Mini-InceptionNet** from scratch. The project demonstrates core computer vision concepts, robust training methodologies, and model deployment via a web interface.

## 🚀 Key Features
* **Custom Architecture**: Built a lightweight version of the Inception network featuring parallel 1x1, 3x3, and 5x5 convolutional branches.
* **Robust Training Pipeline**: Implemented Data Augmentation, `CosineAnnealingLR` scheduler, and SGD with Momentum & Weight Decay.
* **Overfitting Prevention**: Integrated **Model Checkpointing** and **Early Stopping** mechanisms to capture the optimal weights.
* **Experiment Tracking**: Utilized TensorBoard for real-time loss and accuracy visualization.
* **Interactive Web UI**: Deployed the trained model using Gradio for real-time inference.

## 📂 Repository Structure
```text
├── dataset.py         # Data loading and augmentation pipeline
├── model.py           # Mini-InceptionNet architecture definition
├── train.py           # Training loop with Early Stopping & TensorBoard
├── evaluate.py        # Model evaluation and Confusion Matrix generation
├── app.py             # Gradio web interface for real-time inference
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
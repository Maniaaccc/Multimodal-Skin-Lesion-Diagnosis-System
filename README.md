# Multimodal Skin Lesion Diagnosis System

> **A Deep Learning Framework Integrating Dermoscopic Imagery and Clinical Narratives for Enhanced Skin Cancer Detection**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research_Prototype-green?style=for-the-badge)

## Abstract

Dermatological diagnosis is inherently multimodal; clinicians rely on both visual inspection and patient history to differentiate between morphologically similar lesions. Traditional Computer-Aided Diagnosis (CAD) systems often fail to capture this context, leading to misclassification of "visual mimics" (e.g., Early Melanoma vs. Benign Keratosis).

This project presents a **Multimodal Integrated Framework** that fuses **ResNet50-processed visual data** with **Bi-LSTM-processed clinical narratives**. By mimicking the holistic diagnostic workflow of a dermatologist, our proposed fusion model achieves a significant performance leap over unimodal baselines.

## Dataset: ISIC 2019

The system is trained on the **ISIC 2019** dataset, utilizing **25,447 dermoscopic images** across 8 diagnostic categories.

| Class Abbreviation | Diagnosis | Description |
| :--- | :--- | :--- |
| **MEL** | Melanoma | Malignant skin cancer (High Priority) |
| **NV** | Melanocytic Nevus | Common mole (Benign) |
| **BCC** | Basal Cell Carcinoma | Common form of skin cancer |
| **AK** | Actinic Keratosis | Pre-cancerous skin patch |
| **BKL** | Benign Keratosis | Solar lentigo / Seborrheic keratosis |
| **DF** | Dermatofibroma | Benign skin lesion |
| **VASC** | Vascular Lesion | Blood vessel abnormalities |
| **SCC** | Squamous Cell Carcinoma | Second most common skin cancer |

## System Architecture

The model employs a **Late Fusion** strategy combining two distinct deep learning streams:

### 1. Visual Stream (CNN)
* **Backbone:** **ResNet50** (Pre-trained on ImageNet).
* **Technique:** Transfer Learning with fine-tuning (layers unfreezed past layer 100).
* **Preprocessing:** Images resized to $224 \times 224$, normalized, and augmented.

### 2. Textual Stream (RNN)
* **Input:** Synthetic Patient Clinical Narratives (modeling symptoms like "bleeding", "itchy", "growing").
* **Architecture:** **Bi-LSTM** (Bidirectional Long Short-Term Memory) with Attention mechanisms.
* **Innovation:** Implements a **"Text Dropout"** regularization technique to prevent the model from over-relying on text when visual cues are sufficient.

### 3. Fusion Module
* Features from the Global Average Pooling layer (Visual) and Attention layer (Text) are concatenated.
* Passed through a dense classification head to predict the probability distribution across the 8 classes.

## Key Results

The multimodal approach demonstrates superior performance compared to using images alone, particularly in sensitivity for malignant classes.

| Metric | Visual-Only (ResNet50) | **Proposed Multimodal Fusion** |
| :--- | :---: | :---: |
| **Accuracy** | ~76.58% | **96.29%** |
| **Melanoma Recall** | ~66.00% | **96.00%** |
| **F1-Score (Macro)** | ~0.68 | **0.95** |

> **Impact:** The system achieved a **96% Recall for Melanoma**, drastically reducing false negatives for the most dangerous skin cancer type.

## Installation & Usage

### Prerequisites
* Python 3.x
* Google Colab (Recommended for GPU support)

### Dependencies
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

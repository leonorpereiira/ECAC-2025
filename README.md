# Human Activity Recognition (HAR)
### Feature Engineering, Data Augmentation, Deep Learning Embeddings and Experimental Evaluation

This project implements a full machine learning pipeline for **Human Activity Recognition (HAR)** using wearable sensor data from the FORTH-TRACE benchmark dataset.

The work follows a structured experimental methodology, comparing classical feature engineering approaches against deep learning embeddings, and evaluating different preprocessing, splitting, and modeling strategies under rigorous statistical validation.

---

## Objective

The goal of this project is to build and evaluate a **robust activity recognition system** for classifying human physical activities from multimodal sensor data.

The pipeline explores:

- Feature engineering from raw inertial signals
- Synthetic data augmentation (SMOTE)
- Deep learning embeddings via transfer learning
- Within-subject vs between-subject generalization
- Dimensionality reduction and feature selection
- Statistical hypothesis testing of model performance

---

## Dataset

The dataset is based on the **FORTH-TRACE benchmark**, containing multimodal wearable sensor signals collected from multiple participants performing daily activities.

Each sample includes:

- Accelerometer (x, y, z)
- Gyroscope (x, y, z)
- Magnetometer (x, y, z)
- Timestamp
- Activity label (1–7 used in this study)

Activities include:

- Standing
- Sitting
- Walking
- Running
- Stair climbing and transitions
- Combined motion + speech activities

---

## Data Preprocessing

### Windowing Strategy

- 5-second sliding windows
- 50% overlap
- Fixed-length segmentation per subject

### Signal Processing

- Normalization using Z-score
- Outlier analysis (optional exploration stage)
- Sensor fusion across 9 signal channels

---

## Data Augmentation (SMOTE)

To address class imbalance across activities:

- SMOTE (Synthetic Minority Over-sampling Technique) was applied
- Synthetic samples generated in feature space using k-nearest neighbors interpolation
- Used only within training data to avoid data leakage

This improved representation of under-represented activities.

---

## Feature Representations

Two alternative representations were evaluated.

### 1. Handcrafted Feature Set

Extracted from time and frequency domains:

- RMS, mean, variance
- Zero-crossing and mean-crossing rates
- Interquartile range
- Signal energy
- Pairwise axis correlations
- Spectral entropy (FFT-based)
- Dominant frequency components

Computed over sliding windows for all sensor modalities.

---

### 2. Deep Learning Embeddings

Raw accelerometer signals were:

- Resampled to 30Hz (150 timesteps per window)
- Processed using a pretrained **Harnet5 model (SSL-Wearables / OxWearables)**
- Forward pass used without classification head

Result:

- 512-dimensional embedding per window
- Transfer learning from self-supervised wearable representation learning

---

## Feature Processing Pipeline

All experiments include variations of:

- Full feature space
- PCA (90% variance retention)
- ReliefF feature selection (top 15 features)

All transformations were fitted only on training data to prevent leakage.

---

## Data Splitting Strategies

### 1. Within-Subject (Intra-subject)

- 60% train / 20% validation / 20% test per participant
- Same individuals appear in all splits
- Captures personalized motion patterns

### 2. Between-Subject (Inter-subject)

- Split by participant identity:
  - 9 subjects train
  - 3 validation
  - 3 test
- Strict generalization to unseen individuals

This represents the most realistic deployment scenario.

---

## Machine Learning Model

### k-Nearest Neighbors (k-NN)

- Implemented from scratch and validated using scikit-learn
- Hyperparameter tuning: k ∈ {1, 3, 5, 7, 9, 11}
- Selection based on validation F1-macro score

---

## Experimental Design

Each configuration combines:

- Dataset type:
  - Handcrafted Features
  - Deep Embeddings

- Split strategy:
  - Within-subject
  - Between-subject

- Feature processing:
  - Full space
  - PCA
  - ReliefF selection

Total: **12 experimental scenarios**

---

## Results Summary

| Model | Scenario | k | Accuracy | F1-Macro |
|------|----------|--:|----------:|----------:|
| Best model | Features + Within + Full | 1 | 0.7673 | 0.7396 |
| Strong baseline | Features + Within + PCA | 1 | 0.7618 | 0.7349 |
| Embeddings | Within + Full | 1 | 0.6920 | 0.6532 |
| Between-subject best | Features + PCA | 5 | 0.6390 | 0.5870 |
| Worst case | Embeddings + ReliefF | 5 | 0.4858 | 0.4027 |

---

## Key Findings

### 1. Handcrafted features outperform embeddings
Classical signal-based features consistently outperform pretrained embeddings without task-specific fine-tuning.

---

### 2. Generalization gap is significant
Between-subject evaluation shows a substantial drop in performance, confirming strong subject-specific bias in wearable data.

---

### 3. Evaluation strategy strongly affects optimal k

- Within-subject: optimal k = 1
- Between-subject: higher k (5–9) improves robustness

---

### 4. PCA is effective, ReliefF is too aggressive
- PCA preserves structure while reducing dimensionality
- ReliefF removes informative dynamics, reducing performance

---

## Statistical Evaluation

To ensure robust conclusions:

- Friedman test used to compare multiple models
- Nemenyi post-hoc analysis applied for pairwise comparisons
- Results confirmed statistically significant differences between pipelines

---

## Error Analysis

### Activity difficulty levels

**Easy**
- Standing
- Sitting
- Basic transitions

**Medium**
- Walking and stair movements

**Hard**
- Stair climbing + talking (highest confusion due to noisy motion signals)

---

## Deployment Pipeline

The final system includes a full inference pipeline:

Input:
- Raw sensor window (256 × 9 array)

Processing:
- Normalization
- Feature extraction / embedding generation
- PCA / feature selection (if applicable)
- Classification via k-NN

Output:
- Predicted activity label

---

## Technologies

- Python
- NumPy & Pandas
- Scikit-learn
- PyTorch (Harnet5 embeddings)
- SciPy
- Matplotlib & Seaborn
- skrebate (ReliefF)
- scikit-posthocs

---

## Main Topics

- Human Activity Recognition (HAR)
- Time-Series Machine Learning
- Data Augmentation (SMOTE)
- Feature Engineering
- Deep Learning Embeddings
- Transfer Learning
- Dimensionality Reduction (PCA)
- Feature Selection (ReliefF)
- Subject-Independent Evaluation
- Statistical Hypothesis Testing
- Model Deployment Pipelines

---

## Conclusion

This project demonstrates that classical signal engineering combined with structured evaluation pipelines can outperform deep embeddings in HAR tasks when no task-specific fine-tuning is applied. It also highlights the importance of evaluation strategy in estimating real-world performance.

---

# Naive Bayes Classifier, Feature Selection & PCA from Scratch

A complete classification pipeline exploring **Naive Bayes**, **feature selection**, and **PCA (implemented from scratch)** across categorical and numerical datasets.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch%20PCA-orange.svg)

---

## Table of Contents
- [Objective](#objective)
- [Datasets](#datasets)
- [Experimental Pipeline](#experimental-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Team](#team)
- [License](#license)

---

## Objective

Understand how **Naive Bayes** works, explore **feature selection** techniques, and study **dimensionality reduction using PCA** (built from scratch with NumPy). Compare their effects on model performance across categorical and numerical datasets.

**Constraint:** PCA is implemented **entirely from scratch** — `sklearn.decomposition.PCA` is **NOT used**.

---

## Datasets

| Dataset | Type | Source |
|---------|------|--------|
| **Dataset 1 (Mushroom)** | Categorical / Discrete | [UCI ML Repository (ID 73)](https://archive.ics.uci.edu/dataset/73/mushroom) |
| **Dataset 2 (Digits)** | Numerical / Continuous | [Scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |

---

## Experimental Pipeline

Each dataset goes through **three experiments**:

| Experiment | Description | Features Used |
|------------|-------------|---------------|
| **Baseline** | Naive Bayes on all original features | All features |
| **Experiment A** | Feature Selection (ANOVA) → Naive Bayes | Selected subset of original features |
| **Experiment B** | PCA (from scratch) → Naive Bayes | Top-k principal components |

### Evaluation Metrics
- Accuracy
- Classification Report (Precision, Recall, F1)
- Confusion Matrix

### Visualizations
- Confusion matrices (heatmaps)
- Data plotting with PCA and Gaussian PDF contours

---

## Project Structure

```text
Bayes-PCA-Classifier/
├── .gitignore
├── README.md
├── main.py                             # Main execution script to run experiments
├── Experiment_A.py                     # Feature Selection (ANOVA) + Naive Bayes
├── Experiment_B.py                     # PCA from scratch + Naive Bayes
├── naive_bayes_classifier_model.py     # Naive Bayes Classifier Implementation
├── PCA.py                              # PCA from scratch (NumPy only)
├── data_loading.py                     # Dataset loading and preprocessing
└── plotting.py                         # Visualizations and plots
```

---

## Installation

```bash
git clone https://github.com/NimaGalal/Bayes-PCA-Classifier.git
cd Bayes-PCA-Classifier

# Optional: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install numpy pandas scikit-learn matplotlib ucimlrepo
```

---

## Usage

You can run the full pipeline through `main.py`. Inside `main.py`, you can uncomment either `experimentOnNumericalFeatures()` or `experimentOnCategoricalFeatures()` depending on the dataset you wish to evaluate.

```bash
python main.py
```

Or you can import and use specific modules directly in your own scripts:
```python
import data_loading
import Experiment_A
import Experiment_B

# Load numerical data
X_train_num, X_test_num, Y_train_num, Y_test_num = data_loading.dataLoadingAndExtractingNumericalFeatures()

# Run Experiment A (Feature Selection)
Experiment_A.runExperimentA(X_train_num, X_test_num, Y_train_num, Y_test_num)

# Run Experiment B (PCA)
Experiment_B.runExperimentB(X_train_num, X_test_num, Y_train_num, Y_test_num)
```

---

## Results

*To be populated upon executing the experiments.*

The experiments will output the following for each dataset:
- **Baseline Accuracy**: Accuracy of Naive Bayes on the original dataset.
- **Experiment A Accuracy**: Accuracy using top selected features (ANOVA).
- **Experiment B Accuracy**: Accuracy using the top-k components derived via PCA.


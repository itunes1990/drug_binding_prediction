# 🧬 Drug-Target Binding Affinity Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project for predicting drug-target binding affinity using the KIBA (Kinase Inhibitor BioActivity) dataset methodology.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview

This project demonstrates how to:
- Build a synthetic drug-target binding affinity dataset
- Engineer molecular and protein features
- Train a Random Forest model for affinity prediction
- Evaluate model performance using MSE, RMSE, MAE, and R²

## How to use

### 1. Clone the repository
```bash
git clone https://github.com/itunes1990/drug-target-binding-prediction.git
cd drug-target-binding-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the prediction script
```bash
python pkpd_prediction.py
```

### 4. Or use the Jupyter notebook
```bash
jupyter notebook pkpd_example.ipynb
```

## Project Structure

```
drug-target-binding-prediction/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── binding_prediction.py        # Main prediction script
├── example.ipynb        # Interactive notebook example
└── output/                   # Generated results (auto-created)
```

## Methodology

### Step 1: Data Generation
Generate synthetic drug and target features that mimic real molecular properties.

**07 Drug Features:**
| Feature | Description | Unit |
|---------|-------------|------|
| Molecular_Weight | Size of molecule | Daltons |
| LogP | Lipophilicity | - |
| NumAtoms | Heavy atom count | count |
| NumRings | Ring structures | count |
| Polar_Surface_Area | Polar surface | Å² |
| Num_HB_Donors | H-bond donors | count |
| Num_HB_Acceptors | H-bond acceptors | count |

**07 Target Features:**
| Feature | Description | Unit |
|---------|-------------|------|
| Protein_Length | Amino acid count | aa |
| Hydrophobicity | GRAVY score | - |
| Instability_Index | Stability measure | - |
| Isoelectric_Point | pI value | pH |
| Net_Charge | Overall charge | - |
| Num_Domains | Functional domains | count |
| Active_Site_Volume | Binding pocket size | Å³ |

### Step 2: Feature Engineering
Combine drug and target features for each interaction pair.

```python
# Merge features for each drug-target pair
combined_features = drug_features + target_features  # 14 total features
```

### Step 3: Model Training
Train a Random Forest Regressor on 80% of the data.

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
```

### Step 4: Evaluation
Evaluate on 20% held-out test data.

## 📊 Results

### Model Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| MSE | ~0.25 | Mean Squared Error |
| RMSE | ~0.50 | Root Mean Squared Error |
| MAE | ~0.35 | Mean Absolute Error |
| R² | ~0.40 | Coefficient of Determination |

### Top 5 Feature Importance
1. Active_Site_Volume - Binding pocket size
2. Molecular_Weight - Drug molecule size
3. LogP - Drug lipophilicity
4. Protein_Length - Target protein size
5. Hydrophobicity - Target hydrophobic character

## Application

### Basic Usage
```python
from binding_prediction import DrugTargetPredictor

# Initialize predictor
predictor = DrugTargetPredictor()

# Generate and train on synthetic data
predictor.generate_data()
predictor.train()

# Evaluate model
metrics = predictor.evaluate()
print(f"R² Score: {metrics['r2']:.4f}")
```

### Custom Prediction
```python
# Predict binding affinity for new drug-target pairs
drug_features = [350, 2.5, 45, 3, 70, 2, 5]  # 7 drug features
target_features = [450, 0.1, 35, 7, 2, 3, 1500]  # 7 target features

prediction = predictor.predict_single(drug_features, target_features)
print(f"Predicted KIBA Score: {prediction:.3f}")
```

## Visualization Examples

The project generates several visualizations:
- Distribution of binding affinity scores
- Drug property distributions
- Target property distributions
- Feature importance plot
- Actual vs Predicted scatter plot

## For Real KIBA Data

To use the actual KIBA dataset from Therapeutics Data Commons:

```python
from tdc.multi_pred import DTI

# Load real KIBA dataset
data = DTI(name='KIBA')
split = data.get_split()

train_data = split['train']
valid_data = split['valid']
test_data = split['test']
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

Author: Anh Vu Tran.  
For questions or feedback, please open an issue on GitHub.

---
⭐ If you find this project useful, please consider giving it a star!







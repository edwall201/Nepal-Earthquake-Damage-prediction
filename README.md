# Nepal-Earthquake-Damage-prediction

This repository contains machine learning models for predicting earthquake damage to buildings in Nepal. The project uses a mix of supervised learning model comparisons and semi-supervised robustness checks to classify building damage (grades 1-3) based on construction and location features.

## Data Acquisition & Preprocessing

The dataset used for this project is from the **DrivenData competition**: [Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/).

**Note:** The required dataset files (`train_values.csv` and `train_labels.csv`) are already included in the `data/` directory of this repository, so no additional download is necessary.

**Preprocessing details:**
Data preprocessing is handled internally within both scripts (`model_comp.py` and `semi_supervised_learning.py`). The initial preprocessing steps (handled by the `preprocess_data` functions) include:

- Dropping identifier columns (`building_id`).
- Shifting target labels (`damage_grade`) from 1-3 to 0-2 for algorithm compatibility (e.g., XGBoost).
- One-hot encoding of all categorical variables.

To prevent data leakage, Standard scaling (Z-score normalization) of numerical features is strictly applied dynamically _after_ the dataset is split into training and testing/unlabeled sets (using a `Pipeline` in `model_comp.py` and explicit post-split scaling in `semi_supervised_learning.py`).

## How to Run the Code

The project is split into two primary execution scripts:

1. **Supervised Model Comparison (`model_comp.py`)**:
   Runs a comprehensive `GridSearchCV` over 5 different models (KNN, Logistic Regression, Random Forest, XGBoost, Neural Network) and outputs comparison tables, confusion matrices, and ROC/PR curves to the `report/` directory.

   ```bash
   python model_comp.py
   ```

2. Robustness Analysis  
   2.1 Structural & Geographic Analysis

   - **Structural Robustness (`robustness_structure.py`)**:
     Performs subgroup analyses on the held-out test set to ensure the model performs reliably across different physical building characteristics. It evaluates weighted F1 and severe damage recall across foundation Types and Superstructure Material Tiers (Traditional, Mixed/Transitional, Modern). Outputs `foundation_combined.png` and `material_combined.png`.
     ```bash
     python robustness_structure.py
     ```
   - **Geographic Robustness (`robustness_geo.py`)**:
     Analyzes the spatial generalization of the model through two checks:

     1. **Geographic Feature**: Systematically removes geographic features (district, sub-district, ward) to quantify their impact on the Weighted F1-score, outputting `geo_ablation_xgb.png`.
     2. **District-Level Subgroup Analysis**: Evaluates the Weighted F1-score across individual geographic regions (`geo_level_1_id`) on the held-out test set, outputting `geo_level1_subgroup.png`.

     ```bash
     python robustness_geo.py
     ```

     2.2 Semi-Supervised Learning, SHAP, & PCA (`semi_supervised_learning.py`)
     Executes two main pipelines to evaluate learning efficiency and interpretability (all results are saved to the `report/` directory):

      - **Sensitivity Analysis**: Evaluates how model performance (Accuracy, ARI, NMI) scales with the proportion of labeled training data (from 5% to 50%), identifying the optimal "elbow point" of efficiency. Outputs `learning_curve.png`.
      - **Semi-Supervised & Interpretability Analysis**:

      - Evaluates metrics and geographic robustness (ARI, NMI, Accuracy by region) on the unlabeled set, outputting `semi_supervised_results.png` and `semi_supervised_geo_analysis.csv`.
      - Extracts global feature importance using **SHAP**, outputting a bar plot to `shap_summary.png`.
      - Performs dimensionality reduction via **PCA** on the top 10 SHAP-selected features to visualize feature space stability across damage grades, outputting `pca_projection_k10.png`.
      - Generates a summary text report at `semi_supervised_summary.txt`.

      ```bash
      python semi_supervised_learning.py
      ```

3. Application

#### SHAP Analysis

- Extracts the best trained model and scaler from the pipeline
- Computes SHAP values based on the trained XGBoost model
- Separates SHAP values by class and computes global SHAP values by averaging across samples
- Constructs transformed versions of geographic variables (geo_level_1_id, geo_level_2_id, geo_level_3_id) by replacing each category with the mean observed damage level within that category (using training data)
- Applies the same mapping to the test data and fills unseen categories with the global mean
- Uses the transformed variables for interpretation and feature effect analysis

#### GAM

- Reloads and preprocesses the dataset (merge, drop identifiers, relabel target)
- Splits data into training and testing sets with stratification
- Constructs binary target variables: Y ≥ 2, Y ≥ 3
- Applies one-hot encoding to all features
- Identifies feature types: one-hot encoded variables are treated as linear terms; continuous variables are modeled using spline terms
- Automatically builds GAM term structure based on feature types
- Fits two Logistic GAM models separately for: Y ≥ 2, Y ≥ 3 (the final report uses Y ≥ 2)
- Samples a subset of the training data for computational efficiency
- Selects top features for visualization (based on global SHAP in the previous section)
- Computes partial dependence for selected features
- Plots marginal effects of each feature across its value range
  ```bash
  application.ipynb
  ```

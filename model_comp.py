import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,classification_report, confusion_matrix, accuracy_score,
                             f1_score, roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# step 1: Load data
def load_and_prepare_data(X_path, y_path):
    print("="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80)
    # Load the dataset
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    df = X.merge(y, on='building_id')

    print(f"Total samples: {len(df)}")
    print(f"Features: {len(X.columns) - 1}")
    print(f"\nDamage grade distribution:")
    print(df['damage_grade'].value_counts().sort_index())
    print()
    
    return df


# step 2: preprocess data
def preprocess_data(df):
    print("="*80)
    print("STEP 2: PREPROCESSING DATA")
    print("="*80)

    # drop building_id (identifier) and damage_grade (label) from features
    X = df.drop(['building_id', 'damage_grade'], axis=1)
    y = df['damage_grade']
    building_ids = df['building_id']
    geo_level_1 = df['geo_level_1_id'].copy()

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")

    # Encode categorical variables with one hot encoding
    # avoids artificial ordinal relationships that label encoding can introduce
    X_encoded = X.copy()
    
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=False  # set to True if you want fewer columns
    )

    print(f"Original feature count: {X.shape[1]}")
    print(f"Encoded feature count: {X_encoded.shape[1]}")

    # Splitting data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print("Preprocessing complete.")
    print()

    return X_train, y_train, X_test, y_test, building_ids, geo_level_1


# step 3: define models and param grids for comparison
def get_models():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    models = {

        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7, 9, 11]}),

        "Logistic Regression": (LogisticRegression(max_iter=1000), {"model__C": [0.01, 0.1, 1, 10]}),

        "Random Forest": (RandomForestClassifier(random_state=RANDOM_SEED), 
                          {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]}),

        "Neural Network": (MLPClassifier(max_iter=1000, random_state=RANDOM_SEED), 
                           {"model__hidden_layer_sizes": [(50,), (100,), (50, 50)], 
                            "model__alpha": [0.0001, 0.001], "model__learning_rate_init": [0.001, 0.01]})
    }
    return models


# step 4: train models and compare results
def model_comp(X_train, y_train, models):
    '''
    X_train and y_train: training data and labels,
    models: dictionary of models to be tuned, 
        keys are model name, values are tuples of model instance and hyperparameter grid, e.g.:
        {
            "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}), etc
        }
    '''
    print("="*80)
    print("MODEL TRAINING + HYPERPARAMETER TUNING")
    print("="*80)

    # define split to reuse for all models for fair comparison
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}

    for name, (model, param_grid) in models.items():
        print(f"Training {name}...")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=cv, # predefined split for fair comparison
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        results[name] = {
            'best_model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'best_score': grid.best_score_
        }
    
    return results


# step 5: save detailed results and return best model for test evaluation
def save_detailed_results(results, out='report/model_results.txt'):
    with open(out, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON RESULTS (GridSearchCV)\n")
        f.write("="*80 + "\n\n")

        # sort models by best score
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["best_score"],
            reverse=True
        )

        for name, info in sorted_models:
            f.write(f"Model: {name}\n")
            f.write("-"*40 + "\n")
            f.write(f"Best CV Score: {info['best_score']:.5f}\n")
            f.write(f"Best Parameters:\n")

            for param, value in info["best_params"].items():
                f.write(f"  {param}: {value}\n")

            f.write("\n")

    return sorted_models

# step 6: visualize and evaluate on test data
def evaluate_model(sorted_models, X_test, y_test, out='report/model_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) model comparison results
    df = pd.DataFrame([
        {
            "Model": name,
            "CV_F1": info["best_score"]
        }
        for name, info in sorted_models
    ])
    axes[0, 0].barh(df["Model"], df["CV_F1"])
    axes[0, 0].set_title("Model Comparison (CV F1 Score)")
    axes[0, 0].set_xlabel("CV F1 Score")

    # evaluate best model on test data
    best_name, best_info = sorted_models[0]
    best_model = best_info["best_model"]

    y_pred = best_model.predict(X_test)

    #(b) confusion matrix for best model
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    labels = np.array([
        [f"{cm[i,j]}\n({cm_norm[i,j]:.2f})" for j in range(3)]
        for i in range(3)
    ])

    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=axes[0, 1],
                xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    axes[0, 1].set_title(f"Confusion Matrix ({best_name})")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("True")

    # binarize test labels for ROC curve
    y_test_bin = label_binarize(y_test, classes=[1, 2, 3])
    y_score = best_model.predict_proba(X_test)

    # (c) roc
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, label=f"Class {i+1} (AUC={roc_auc:.2f})")

    axes[1, 0].plot([0, 1], [0, 1], "k--")
    axes[1, 0].set_title("ROC Curve (Random Forest, OvR)")
    axes[1, 0].legend()
    axes[1,0].grid()
    axes[1,0].set_xlabel("False Positive Rate")
    axes[1,0].set_ylabel("True Positive Rate")

    # (d) precision-recall curve
    for i in range(3):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )
        axes[1, 1].plot(recall, precision, label=f"Class {i+1}")
        pos_rate = np.mean(y_test_bin[:, i])
        axes[1, 1].axhline(pos_rate, linestyle="--", color='gray', alpha=0.6)

    axes[1, 1].set_title("Precision–Recall Curve (Random Forest, OvR)")
    axes[1, 1].legend()
    axes[1,1].grid()
    axes[1,1].set_xlabel("Recall")
    axes[1,1].set_ylabel("Precision")

    
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')

def run_model_comp(X_path, y_path):
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")

    # step 1: load data
    df = load_and_prepare_data(X_path, y_path)

    # step 2: preprocess data
    X_train, y_train, X_test, y_test, building_ids, geo_level_1 = preprocess_data(df)

    # step 3: define models and param grids for comparison
    models = get_models()

    # step 4: train models and compare results
    # results = model_comp(X_train, y_train, models)
    
    # # save results for reload and analysis later
    # with open('report/model_comparison_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # reloading results from pickle file for analysis
    with open('report/model_comparison_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # step 5: save detailed results and return best model for test evaluation
    sorted_models = save_detailed_results(results)

    # step 6: visualize and evaluate on test data
    # evaluate_model(sorted_models, X_test, y_test)

    




    print('pause...')
    


if __name__ == "__main__":
    run_model_comp(X_path="data/train_values.csv", 
                   y_path="data/train_labels.csv")
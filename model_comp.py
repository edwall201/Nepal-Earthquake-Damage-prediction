import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,classification_report, confusion_matrix, accuracy_score,
                             f1_score, roc_curve, auc, precision_recall_curve, cohen_kappa_score)
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import make_scorer
from sklearn.base import clone

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
    y = df['damage_grade'] - 1
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

    # get class weights for rebalancing the cost function during model comparison
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    weights = {
        c: total / (n_classes * cnt)
        for c, cnt in zip(classes, counts)
    }
    # normalize weights
    mean_w = np.mean(list(weights.values()))
    weights = {k: v / mean_w for k, v in weights.items()}
    
    print("Preprocessing complete.")
    print()

    return X_train, y_train, X_test, y_test, building_ids, geo_level_1, weights


# step 3: define models and param grids for comparison
def get_models():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    models = {

        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7, 9, 11]}),

        "Logistic Regression": (LogisticRegression(max_iter=1000), {"model__C": [0.01, 0.1, 1, 10]}),

        "Random Forest": (RandomForestClassifier(random_state=RANDOM_SEED), 
                          {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]}),

         "XGBoost": (XGBClassifier(objective="multi:softprob",num_class=3, eval_metric="mlogloss",random_state=RANDOM_SEED),
                     {"model__n_estimators": [100, 200, 300], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1]}),

        "Neural Network": (MLPClassifier(max_iter=1000, random_state=RANDOM_SEED), 
                           {"model__hidden_layer_sizes": [(25,25), (50, 50)], 
                            "model__alpha": [0.0001, 0.001], "model__learning_rate_init": [0.001, 0.01]})
    }
    return models


def create_cost(weights):
    def cost_function(y_true, y_pred):
        total = 0

        for t, p in zip(y_true, y_pred):
            cost = 1 if t !=p else 0
            total += weights[int(t)] * cost
        return -(total / len(y_true))

    return make_scorer(cost_function)


# step 4: train models and compare results
def model_comp(X_train, y_train, models, weights=None):
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

    if weights is not None:
        custom_scorer = {
            "f1": make_scorer(f1_score, average='weighted'),
            "cost": create_cost(weights),
            "qwk": make_scorer(cohen_kappa_score, weights="quadratic")
        }
    else:
        custom_scorer = {
            "f1": make_scorer(f1_score, average='weighted'),
            "qwk": make_scorer(cohen_kappa_score, weights="quadratic")
        }

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
            scoring=custom_scorer, 
            refit="cost", # custom cost function more heavily penalize misclassification
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        results[name] = {
            'best_model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'best_cost': grid.best_score_,
            'cv_results': grid.cv_results_,
            'best_qwk': grid.cv_results_['mean_test_qwk'][grid.best_index_],
            'best_f1': grid.cv_results_['mean_test_f1'][grid.best_index_],
            'refit_metric': 'cost'
        }
    
    return results


# step 5: save detailed results and return best model for test evaluation
def save_detailed_results(results, out='report/model_results.txt'):
    summary = {}

    for name, info in results.items():

        cv_res = info['cv_results']
        best_idx = np.argmax(cv_res['mean_test_f1'])

        summary[name] = {
            "cost": info["best_cost"],
            "qwk": info["best_qwk"],
            "f1": info["best_f1"],
            "f1_std": cv_res['std_test_f1'][best_idx],
            "best_params": info["best_params"],
            "best_model": info["best_model"]
        }

    # sort by cost (since that's your refit metric)
    sorted_models = sorted(
        summary.items(),
        key=lambda x: x[1]["cost"],
        reverse=True
    )

    with open(out, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON RESULTS (GridSearchCV)\n")
        f.write("Refit metric: Weighted Cost\n")
        f.write("="*80 + "\n\n")

        for name, info in sorted_models:
            f.write("-"*40 + "\n")
            f.write(f"Model: {name}\n")
            f.write("-"*40 + "\n")

            f.write(f"Weighted Cost: {info['cost']:.4f}\n")
            f.write(f"Weighted F1:    {info['f1']:.4f}\n")
            f.write(f"Quadratic Kappa:{info['qwk']:.4f}\n\n")

            f.write("Best Parameters:\n")
            for param, value in info["best_params"].items():
                f.write(f"  {param}: {value}\n")

            f.write("\n")

    return sorted_models


def format_param_grid(param_grid):
    parts = []
    for k, v in param_grid.items():
        name = k.replace("model__", "")
        parts.append(f"{name}: {v}")
    return "\n".join(parts)


def format_best_params(best_params):
    parts = []
    for k, v in best_params.items():
        name = k.replace("model__", "")
        parts.append(f"{name}={v}")
    return "\n".join(parts)


def plot_model_comps(sorted_models, out='report/model_comparison.png'):
    # for tuning record
    models = get_models()
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.axis("off")

    table_data = []

    for name, info in sorted_models:
        param_grid = models[name][1]

        search_space = format_param_grid(param_grid)
        best_params = format_best_params(info["best_params"])

        # score = f"{info['cost']:.3f}"
        f1_mean = f"{info['f1']:.3f}"
        f1_std = f"{info['f1_std']:.4f}"
        qwk = f"{info['qwk']:.3f}"

        f1 = f"{info['f1']:.3f}±{info['f1_std']:.4f}"

        table_data.append([name, search_space, best_params, f1, qwk])

    table = ax.table(
        cellText=table_data,
        colLabels=[
            "Model",
            "Hyperparameters Searched",
            "Best Parameters",
            "Weighted F1\n(mean ± std)",
            # "Cost",
            "QWK"
        ],
        cellLoc="center",
        loc="center"
    )

    col_widths = {
        0: 0.11,  # Model
        1: 0.25,  # Hyperparameters Searched
        2: 0.23,  # Best Parameters
        # 3: 0.09,  # Cost
        3: 0.15,  # F1
        4: 0.09   # QWK
    }

    for (row, col), cell in table.get_celld().items():

        if row == 0:
            cell.set_height(0.10)
        else:
            cell.set_height(0.12)

        cell.set_text_props(wrap=True)
        cell.set_width(col_widths[col])

        if row == 1 and col in [0, 3, 4]:
            cell.set_text_props(weight="bold")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title("Model Comparison (Cross-Validated Performance)", y=1.05)

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# step 6: visualize and evaluate on test data
def evaluate_model(sorted_models, X_test, y_test, out='report/model_analysis.png'):
    fig = plt.figure(figsize=(12, 4)) # was 16, 10
    gs = fig.add_gridspec(1, 3)

    # Best model
    best_name, best_info = sorted_models[0]
    best_model = best_info["best_model"]

    y_pred = best_model.predict(X_test)

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = best_model.predict_proba(X_test)

    ax_cm = fig.add_subplot(gs[0, 0])

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    labels = np.array([
        [f"{cm[i,j]}\n({cm_norm[i,j]:.2f})" for j in range(3)]
        for i in range(3)
    ])

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=[1, 2, 3],
        yticklabels=[1, 2, 3]
    )
    ax_cm.set_aspect("equal")
    ax_cm.set_title(f"Confusion Matrix ({best_name})")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")

    ax_roc = fig.add_subplot(gs[0, 1])

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_best = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"Class {i+1} (AUC={auc_best:.2f})")

    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_title(f"ROC Curve ({best_name}, OvR)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid()

    ax_pr = fig.add_subplot(gs[0, 2])

    for i in range(3):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        ax_pr.plot(recall, precision, label=f"Class {i+1}")

        pos_rate = np.mean(y_test_bin[:, i])
        ax_pr.axhline(pos_rate, linestyle="--", color="gray", alpha=0.6)

    ax_pr.set_title(f"Precision–Recall Curve ({best_name}, OvR)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    ax_pr.grid()

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_classification_report(sorted_models, X_test, y_test, out='report/best_model_report.png'):
    best_name, best_info = sorted_models[0]
    best_model = best_info["best_model"]

    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()

    df_classes = df.loc[["0", "1", "2"], ["precision", "recall", "f1-score"]]
    df_classes.index = ["Class 1", "Class 2", "Class 3"]
    df_classes = df_classes.round(3)

    weighted_f1 = report["weighted avg"]["f1-score"]
    accuracy = report["accuracy"]

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")

    table = ax.table(
        cellText=df_classes.values,
        rowLabels=df_classes.index,
        colLabels=df_classes.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title(
        f"{best_name} Performance\n"
        f"Weighted F1: {weighted_f1:.3f}   |   Accuracy: {accuracy:.3f}",
        pad=0
    )

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def binary_tasks(params, X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier

    # greater than or equal to 2 vs less than 2
    y_train_2 = (y_train.values >= 1).astype(int)
    y_test_2 = (y_test.values >= 1).astype(int)
    model_2 = XGBClassifier(
        objective="binary:logistic",
        n_estimators=params["model__n_estimators"],
        max_depth=params["model__max_depth"],
        learning_rate=params["model__learning_rate"],
        random_state=RANDOM_SEED,
        eval_metric="logloss"
    )
    model_2.fit(X_train, y_train_2)
    pred2 = model_2.predict(X_test)
    print("Y >= 2")
    print(classification_report(y_test_2, pred2))


    # 
    y_train_3 = (y_train.values >= 2).astype(int)
    y_test_3 = (y_test.values >= 2).astype(int)
    model_3 = XGBClassifier(
        objective="binary:logistic",
        n_estimators=params["model__n_estimators"],
        max_depth=params["model__max_depth"],
        learning_rate=params["model__learning_rate"],
        random_state=RANDOM_SEED,
        eval_metric="logloss"
    )
    model_3.fit(X_train, y_train_3)
    pred3 = model_3.predict(X_test)
    print("Y >= 3")
    print(classification_report(y_test_3, pred3))

    final_out = np.zeros_like(pred2)
    mask_wrong = np.zeros_like(pred2, dtype=bool)

    count0 = 0
    count1 = 0
    count2 = 0
    count_wrong = 0

    for i, (p2, p3) in enumerate(zip(pred2, pred3)):
        if p2 == 0 and p3 == 0:
            count0+=1
            final_out[i] = 0
        elif p2 == 0 and p3 == 1:
            count1+=1
            final_out[i] = 1
        elif p2 == 1 and p3 == 1:
            count2+=1
            final_out[i] = 2
        elif p2 == 1 and p3 == 0:
            count_wrong+=1
            # print("BAD")
            final_out[i] = 0
            mask_wrong[i] = True

        final_accuracy = np.mean((final_out == y_test) & (~mask_wrong))

        


def run_model_comp(X_path, y_path):
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")

    # step 1: load data
    df = load_and_prepare_data(X_path, y_path)

    # step 2: preprocess data
    X_train, y_train, X_test, y_test, building_ids, geo_level_1, weights = preprocess_data(df)

    # step 3: define models and param grids for comparison
    models = get_models()

    # step 4: train models and compare results
    results = model_comp(X_train, y_train, models, weights)
    
    # # save results for reload and analysis later
    # with open('report/model_comparison_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # # # reloading results from pickle file for analysis
    # with open(r'C:\Users\akm87\Documents\github\model_comparison_results.pkl', 'rb') as f:
    #     results = pickle.load(f)

    # step 5: save detailed results and return best model for test evaluation
    sorted_models = save_detailed_results(results)

    # step 6: visualize and evaluate on test data
    evaluate_model(sorted_models, X_test, y_test)

    best_params = sorted_models[0][1]['best_params']


    print('Model comparison complete!')
    


if __name__ == "__main__":
    run_model_comp(X_path="data/train_values.csv", 
                   y_path="data/train_labels.csv")
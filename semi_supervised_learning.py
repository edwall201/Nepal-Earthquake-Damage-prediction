import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# STEP 1 & 2: DATA LOADING & PREPROCESSING
# ============================================================================
def load_and_prepare_data(labels_path, values_path):
    labels_df = pd.read_csv(labels_path)
    values_df = pd.read_csv(values_path)
    return values_df.merge(labels_df, on='building_id')

def preprocess_data(df):
    X = df.drop(['building_id', 'damage_grade'], axis=1)
    y = df['damage_grade'] - 1 # XGBoost expects 0, 1, 2
    building_ids = df['building_id']
    geo_level_1 = df['geo_level_1_id'].copy()
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    # One-hot encoding to align with the main model methodology
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X_encoded.index)
    
    return X_scaled_df, y, building_ids, geo_level_1

# ============================================================================
# STEP 7: GEOGRAPHIC ANALYSIS
# ============================================================================
def analyze_geographic_subsets(y_true, y_pred, geo_ids):
    unique_geo_levels = sorted(geo_ids.unique())
    geo_results = []
    
    for geo_id in unique_geo_levels:
        geo_mask = geo_ids == geo_id
        if geo_mask.sum() < 5: continue

        y_t = y_true[geo_mask]
        y_p = y_pred[geo_mask]
        
        geo_results.append({
            'geo_level_1_id': geo_id, 
            'n_samples': len(y_t), 
            'ARI': adjusted_rand_score(y_t, y_p), 
            'NMI': normalized_mutual_info_score(y_t, y_p), 
            'Accuracy': accuracy_score(y_t, y_p)
        })
    
    return pd.DataFrame(geo_results).sort_values('n_samples', ascending=False)

# ===========================================================================
# STEP 8 & 9: VISUALIZATION & SAVING
# ============================================================================

def create_visualizations(y_true, y_pred, metrics, geo_results_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('Robustness Check: Semi-Supervised Learning Results (XGBoost)', fontsize=26, fontweight='bold', y=1.02)
    ax1 = axes[0]
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
    xticklabels=[1, 2, 3], yticklabels=[1, 2, 3],
    annot_kws={"size": 22}, cbar=False)
    ax1.set_title('Confusion Matrix\n(Unlabeled Set: True vs Predicted)', fontsize=20, pad=15)
    ax1.set_xlabel('Predicted Damage Grade', fontsize=18)
    ax1.set_ylabel('True Damage Grade', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    ax2 = axes[1]
    geo_top_15 = geo_results_df.head(15)
    x_geo = range(len(geo_top_15))
    ax2.plot(x_geo, geo_top_15['ARI'], marker='o', label='ARI', linewidth=3, markersize=10)
    ax2.plot(x_geo, geo_top_15['NMI'], marker='s', label='NMI', linewidth=3, markersize=10)
    ax2.plot(x_geo, geo_top_15['Accuracy'], marker='^', label='Accuracy', linewidth=3, markersize=10)
    ax2.axhline(y=metrics['ARI'], color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Overall ARI')
    ax2.axhline(y=metrics['NMI'], color='green', linestyle=':', alpha=0.5, linewidth=2, label='Overall NMI')
    ax2.set_title('Performance Metrics by Geographic Region (Top 15)', fontsize=20, pad=15)
    ax2.set_xlabel('Geographic Region (geo_level_1_id)', fontsize=18)
    ax2.set_ylabel('Score', fontsize=18)
    ax2.set_xticks(x_geo)
    ax2.set_xticklabels(geo_top_15['geo_level_1_id'], rotation=45, fontsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=16)
    ax2.legend(loc='best', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization (Large Font) saved to: {output_path}")

    print()

def save_detailed_results(metrics, geo_results_df, labeled_percentage, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = f"{output_dir}/semi_supervised_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SEMI-SUPERVISED LEARNING ROBUSTNESS CHECK SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Algorithm: XGBoost\n")
        f.write(f"Labeled Data Proportion: {labeled_percentage*100}%\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall Accuracy: {metrics['Accuracy']:.4f}\n")
        f.write(f"Overall ARI:      {metrics['ARI']:.4f}\n")
        f.write(f"Overall NMI:      {metrics['NMI']:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write("Interpretation: ARI and NMI confirm structural consistency of the feature space.\n")
    
    geo_results_df.to_csv(f"{output_dir}/semi_supervised_geo_analysis.csv", index=False)
    print(f"Detailed results saved to: {summary_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_semi_supervised_analysis(labels_path, values_path, labeled_percentage=0.5, output_dir='./report'):
    df = load_and_prepare_data(labels_path, values_path)
    X_scaled_df, y, building_ids, geo_level_1 = preprocess_data(df)
    
    # Step 3: Split into labeled and unlabeled simulation
    X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
        X_scaled_df, y, test_size=(1-labeled_percentage), stratify=y, random_state=RANDOM_SEED
    )
    geo_unlabeled = geo_level_1.loc[X_unlabeled.index]
    
    # Step 4: Training
    print(f"Training XGBoost on {labeled_percentage*100}% labeled data...")
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    model.fit(X_labeled, y_labeled)
    y_pred = model.predict(X_unlabeled)
    
    # Step 5: Metrics Evaluation
    metrics = {
        'ARI': adjusted_rand_score(y_unlabeled_true, y_pred),
        'NMI': normalized_mutual_info_score(y_unlabeled_true, y_pred),
        'Accuracy': accuracy_score(y_unlabeled_true, y_pred),
        'confusion_matrix': confusion_matrix(y_unlabeled_true, y_pred)
    }
    
    geo_results_df = analyze_geographic_subsets(y_unlabeled_true, y_pred, geo_unlabeled)
    
    # Step 6: Output
    create_visualizations(y_unlabeled_true, y_pred, metrics, geo_results_df, f"{output_dir}/semi_supervised_results.png")
    save_detailed_results(metrics, geo_results_df, labeled_percentage, output_dir)
    
    print(f"Execution Complete. Accuracy: {metrics['Accuracy']:.4f}, ARI: {metrics['ARI']:.4f}, NMI: {metrics['NMI']:.4f}")

if __name__ == "__main__":
    run_semi_supervised_analysis(
        labels_path='data/train_labels.csv', 
        values_path='data/train_values.csv'
    )
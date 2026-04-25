import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, accuracy_score)
from sklearn.decomposition import PCA
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# STEP 1 & 2: DATA LOADING & PREPROCESSING
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

# STEP 7: GEOGRAPHIC ANALYSIS
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

# STEP 8 & 9: VISUALIZATION & SAVING
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

#SHAP
def run_shap_analysis(model, X_sample, output_dir):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Top Predictors)", fontsize=18)
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP Summary plot saved to: {output_dir}/shap_summary.png")

#PCA
def run_pca_visualization(X_sample, y_sample, output_dir):
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='viridis', alpha=0.6, s=20)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Damage Grade (0: Low, 1: Medium, 2: High)', fontsize=14)
    
    plt.title("PCA Projection: Latent Feature Space Stability", fontsize=18)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_dir}/pca_projection.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA Projection plot saved to: {output_dir}/pca_projection.png")

# Main execution
def plot_sensitivity_curve(df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Proportion'] * 100, df['Accuracy'], marker='o', label='Accuracy', linewidth=2)
    plt.plot(df['Proportion'] * 100, df['ARI'], marker='s', label='ARI', linewidth=2)
    plt.title('Model Sensitivity to Labeled Data Volume', fontsize=16)
    plt.xlabel('Percentage of Labeled Training Data (%)', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/learning_curve.png")
    
def run_sensitivity_analysis(labels_path, values_path, output_dir='./report'):
    df = load_and_prepare_data(labels_path, values_path)
    X_scaled_df, y, building_ids, geo_level_1 = preprocess_data(df)
    
    # Hold out 20% as a consistent test set to evaluate all variations fairly
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled_df, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    proportions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for p in proportions:
        print(f"Testing labeled proportion: {p*100}%...")
        # Sample p portion of the training data
        X_labeled, _, y_labeled, _ = train_test_split(
            X_train_full, y_train_full, train_size=p, stratify=y_train_full, random_state=RANDOM_SEED
        )
        
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
        model.fit(X_labeled, y_labeled)
        y_pred = model.predict(X_test)
        
        results.append({
            'Proportion': p,
            'Accuracy': accuracy_score(y_test, y_pred),
            'ARI': adjusted_rand_score(y_test, y_pred),
            'NMI': normalized_mutual_info_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    plot_sensitivity_curve(results_df, output_dir)
    results_df = pd.DataFrame(results)
    
    # Calculate Marginal Gain (Difference between current and previous accuracy)
    results_df['Gain'] = results_df['Accuracy'].diff().fillna(0)
    
    # Identify the "Best" based on different criteria
    absolute_best = results_df.loc[results_df['Accuracy'].idxmax()]
    
    # Identify the Elbow Point (where gain drops below a certain threshold, e.g., 1%)
    # This is often the most "robust" model for sparse data scenarios
    threshold = 0.01 
    elbow_point = results_df[results_df['Gain'] >= threshold].iloc[-1] if any(results_df['Gain'] >= threshold) else results_df.iloc[0]

    print(f"Absolute Highest Accuracy: {absolute_best['Accuracy']:.4f} (at {absolute_best['Proportion']*100:.0f}% labels)")
    print(f"Optimal Efficiency Point:  {elbow_point['Accuracy']:.4f} (at {elbow_point['Proportion']*100:.0f}% labels)")
    print(f"Observation: Beyond {elbow_point['Proportion']*100:.0f}%, marginal accuracy gain is less than {threshold*100:.1f}%.")
    return results_df

def run_semi_supervised_analysis(labels_path, values_path, labeled_percentage=0.5, output_dir='./report'):
    df = load_and_prepare_data(labels_path, values_path)
    X_scaled_df, y, building_ids, geo_level_1 = preprocess_data(df)
    
    X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
        X_scaled_df, y, test_size=(1-labeled_percentage), stratify=y, random_state=RANDOM_SEED
    )
    geo_unlabeled = geo_level_1.loc[X_unlabeled.index]
    
    print(f"Training XGBoost on {labeled_percentage*100}% labeled data...")
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    model.fit(X_labeled, y_labeled)
    y_pred = model.predict(X_unlabeled)
    
    metrics = {
        'ARI': adjusted_rand_score(y_unlabeled_true, y_pred),
        'NMI': normalized_mutual_info_score(y_unlabeled_true, y_pred),
        'Accuracy': accuracy_score(y_unlabeled_true, y_pred),
        'confusion_matrix': confusion_matrix(y_unlabeled_true, y_pred)
    }
    
    geo_results_df = analyze_geographic_subsets(y_unlabeled_true, y_pred, geo_unlabeled)
    
    # Run SHAP Analysis with a sample of 2000 rows
    run_shap_analysis(model, X_unlabeled.iloc[:2000], output_dir)
    
    # Run PCA Analysis with a sample of 5000 rows
    run_pca_visualization(X_unlabeled.iloc[:5000], y_unlabeled_true.iloc[:5000], output_dir)
    
    create_visualizations(y_unlabeled_true, y_pred, metrics, geo_results_df, f"{output_dir}/semi_supervised_results.png")
    save_detailed_results(metrics, geo_results_df, labeled_percentage, output_dir)
    
    print(f"Execution Complete. Accuracy: {metrics['Accuracy']:.4f}, ARI: {metrics['ARI']:.4f}, NMI: {metrics['NMI']:.4f}")

if __name__ == "__main__":
    run_sensitivity_analysis(labels_path='data/train_labels.csv', values_path='data/train_values.csv')
    run_semi_supervised_analysis(labels_path='data/train_labels.csv', values_path='data/train_values.csv')

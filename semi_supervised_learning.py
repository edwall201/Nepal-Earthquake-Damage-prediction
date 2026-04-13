import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,classification_report, confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
def load_and_prepare_data(labels_path, values_path):
    print("="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80)
    
    # Load data
    labels_df = pd.read_csv(labels_path)
    values_df = pd.read_csv(values_path)
    
    # Merge datasets
    df = values_df.merge(labels_df, on='building_id')
    
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(values_df.columns) - 1}")
    print(f"\nDamage grade distribution:")
    print(df['damage_grade'].value_counts().sort_index())
    print()
    
    return df


# ============================================================================
# STEP 2: PREPROCESS DATA
# ============================================================================
def preprocess_data(df):
    
    print("="*80)
    print("STEP 2: PREPROCESSING DATA")
    print("="*80)
    
    # Separate features and target
    X = df.drop(['building_id', 'damage_grade'], axis=1)
    y = df['damage_grade']
    building_ids = df['building_id']
    geo_level_1 = df['geo_level_1_id'].copy()
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale features (important for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X_encoded.index)
    
    print("Preprocessing complete.")
    print()
    
    return X_scaled_df, y, building_ids, geo_level_1, label_encoders, scaler


# ============================================================================
# STEP 3: SPLIT INTO LABELED AND UNLABELED
# ============================================================================
def split_labeled_unlabeled(X_scaled_df, y, geo_level_1, labeled_percentage=0.5):
    print("="*80)
    print("STEP 3: SPLITTING INTO LABELED AND UNLABELED")
    print("="*80)
    
    # Stratified split to maintain class distribution
    labeled_indices, unlabeled_indices = train_test_split(
        range(len(X_scaled_df)),
        test_size=(1 - labeled_percentage),
        stratify=y,
        random_state=RANDOM_SEED
    )
    
    # Create labeled and unlabeled datasets
    X_labeled = X_scaled_df.iloc[labeled_indices]
    y_labeled = y.iloc[labeled_indices]
    geo_labeled = geo_level_1.iloc[labeled_indices]
    
    X_unlabeled = X_scaled_df.iloc[unlabeled_indices]
    y_unlabeled_true = y.iloc[unlabeled_indices]  # Ground truth (hidden in semi-supervised setting)
    geo_unlabeled = geo_level_1.iloc[unlabeled_indices]
    
    print(f"Labeled samples: {len(X_labeled)} ({labeled_percentage*100:.1f}%)")
    print(f"Unlabeled samples: {len(X_unlabeled)} ({(1-labeled_percentage)*100:.1f}%)")
    print(f"\nLabeled set damage distribution:")
    print(y_labeled.value_counts().sort_index())
    print(f"\nUnlabeled set (true) damage distribution:")
    print(y_unlabeled_true.value_counts().sort_index())
    print()
    
    labeled_data = {'X': X_labeled, 'y': y_labeled, 'geo': geo_labeled}
    unlabeled_data = {'X': X_unlabeled, 'y_true': y_unlabeled_true, 'geo': geo_unlabeled, 'indices': unlabeled_indices}
    
    return labeled_data, unlabeled_data


# ============================================================================
# STEP 4: TRAIN KNN ON LABELED DATA
# ============================================================================
def train_knn_model(labeled_data, k_values=[3, 5, 7, 9, 11]):
    print("="*80)
    print("STEP 4: TRAINING KNN CLASSIFIER ON LABELED DATA")
    print("="*80)
    
    X_labeled = labeled_data['X']
    y_labeled = labeled_data['y']
    
    best_k = None
    best_score = -1
    
    print("Testing different K values (using 5-fold cross-validation):")
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        scores = cross_val_score(knn, X_labeled, y_labeled, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        print(f"  K={k}: {mean_score:.4f} (+/- {scores.std()*2:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    
    print(f"\nBest K: {best_k} with accuracy {best_score:.4f}")
    
    # Train final model with best K
    knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
    knn_model.fit(X_labeled, y_labeled)
    
    print(f"Model trained with K={best_k}")
    print()
    
    return knn_model, best_k, best_score


# ============================================================================
# STEP 5: ASSIGN PSEUDO-LABELS TO UNLABELED DATA
# ============================================================================
def assign_pseudo_labels(knn_model, unlabeled_data):
    print("="*80)
    print("STEP 5: ASSIGNING PSEUDO-LABELS TO UNLABELED DATA")
    print("="*80)
    
    X_unlabeled = unlabeled_data['X']
    
    # Predict on unlabeled data
    y_unlabeled_pred = knn_model.predict(X_unlabeled)
    
    print(f"Pseudo-labels assigned to {len(y_unlabeled_pred)} samples")
    print(f"\nPredicted label distribution:")
    print(pd.Series(y_unlabeled_pred).value_counts().sort_index())
    print()
    
    return y_unlabeled_pred


# ============================================================================
# STEP 6: EVALUATE LABEL DISTRIBUTIONS
# ============================================================================
def evaluate_label_distributions(y_true, y_pred):
    print("="*80)
    print("STEP 6: EVALUATING LABEL DISTRIBUTIONS")
    print("="*80)
    
    # Calculate metrics
    ari_score = adjusted_rand_score(y_true, y_pred)
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("OVERALL METRICS:")
    print(f"  Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi_score:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print()
    
    # Confusion matrix
    print("Confusion Matrix (rows=true, columns=predicted):")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f'True_{i}' for i in sorted(np.unique(y_true))], columns=[f'Pred_{i}' for i in sorted(np.unique(y_true))])
    print(cm_df)
    print()
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print()
    
    metrics = {'ARI': ari_score, 'NMI': nmi_score, 'Accuracy': accuracy, 'confusion_matrix': cm }
    
    return metrics


# ============================================================================
# STEP 7: ANALYZE WITHIN GEOGRAPHIC SUBSETS
# ============================================================================
def analyze_geographic_subsets(y_true, y_pred, geo_ids):
    print("="*80)
    print("STEP 7: ANALYZING WITHIN GEOGRAPHIC SUBSETS")
    print("="*80)
    
    # Get unique geo levels
    unique_geo_levels = sorted(geo_ids.unique())
    print(f"Number of geographic regions (geo_level_1): {len(unique_geo_levels)}")
    print()
    
    # Calculate metrics for each geographic region
    geo_results = []
    
    for geo_id in unique_geo_levels:
        # Get indices for this geographic region
        geo_mask = geo_ids == geo_id
        
        if geo_mask.sum() < 5:  # Skip regions with too few samples
            continue
        
        y_true_geo = y_true[geo_mask]
        y_pred_geo = y_pred[geo_mask]
        
        # Calculate metrics
        ari_geo = adjusted_rand_score(y_true_geo, y_pred_geo)
        nmi_geo = normalized_mutual_info_score(y_true_geo, y_pred_geo)
        acc_geo = accuracy_score(y_true_geo, y_pred_geo)
        n_samples = len(y_true_geo)
        
        geo_results.append({ 'geo_level_1_id': geo_id, 'n_samples': n_samples, 'ARI': ari_geo, 'NMI': nmi_geo, 'Accuracy': acc_geo })
    
    geo_results_df = pd.DataFrame(geo_results)
    geo_results_df = geo_results_df.sort_values('n_samples', ascending=False)
    
    print("Top 10 geographic regions by sample size:")
    print(geo_results_df.head(10).to_string(index=False))
    print()
    
    print("Summary statistics across geographic regions:")
    print(geo_results_df[['ARI', 'NMI', 'Accuracy']].describe())
    print()
    
    return geo_results_df


# ============================================================================
# STEP 8: CREATE VISUALIZATIONS
# ============================================================================
def create_visualizations(y_true, y_pred, metrics, geo_results_df, output_path):
    print("="*80)
    print("STEP 8: CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Semi-Supervised Learning Results: KNN Pseudo-Labeling',fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix Heatmap
    ax1 = axes[0, 0]
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    ax1.set_title('Confusion Matrix\n(Unlabeled Set: True vs Predicted)')
    ax1.set_xlabel('Predicted Damage Grade')
    ax1.set_ylabel('True Damage Grade')
    
    # Plot 2: Label Distribution Comparison
    ax2 = axes[0, 1]
    x_pos = np.arange(3)
    width = 0.35
    
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    ax2.bar(x_pos - width/2, true_counts.values, width, label='True Labels', alpha=0.8)
    ax2.bar(x_pos + width/2, pred_counts.values, width, label='Predicted Labels', alpha=0.8)
    ax2.set_xlabel('Damage Grade')
    ax2.set_ylabel('Count')
    ax2.set_title('Label Distribution: True vs Predicted\n(Unlabeled Set)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([1, 2, 3])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Performance Metrics by Geographic Region
    ax3 = axes[1, 0]
    geo_top_20 = geo_results_df.head(20)
    x_geo = range(len(geo_top_20))
    
    ax3.plot(x_geo, geo_top_20['ARI'], marker='o', label='ARI', linewidth=2)
    ax3.plot(x_geo, geo_top_20['NMI'], marker='s', label='NMI', linewidth=2)
    ax3.plot(x_geo, geo_top_20['Accuracy'], marker='^', label='Accuracy', linewidth=2)
    ax3.axhline(y=metrics['ARI'], color='blue', linestyle='--', alpha=0.5, label='Overall ARI')
    ax3.axhline(y=metrics['NMI'], color='orange', linestyle='--', alpha=0.5, label='Overall NMI')
    ax3.set_xlabel('Geographic Region (Top 20 by sample size)')
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics by Geographic Region')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Distribution of Metrics Across Regions
    ax4 = axes[1, 1]
    metrics_data = geo_results_df[['ARI', 'NMI', 'Accuracy']].values.flatten()
    metrics_labels = (['ARI']*len(geo_results_df) + ['NMI']*len(geo_results_df) + ['Accuracy']*len(geo_results_df))
    metrics_df_plot = pd.DataFrame({'Metric': metrics_labels, 'Score': metrics_data})
    
    sns.violinplot(data=metrics_df_plot, x='Metric', y='Score', ax=ax4)
    ax4.set_title('Distribution of Metrics Across Geographic Regions')
    ax4.set_ylabel('Score')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved to: {output_path}")
    print()


# ============================================================================
# STEP 9: SAVE DETAILED RESULTS
# ============================================================================
def save_detailed_results(building_ids, unlabeled_indices, y_true, y_pred, geo_ids, geo_results_df, metrics, best_k, labeled_percentage, output_dir):
    
    print("="*80)
    print("STEP 9: SAVING DETAILED RESULTS")
    print("="*80)
    
    # Create comprehensive results dataframe
    results_detailed = pd.DataFrame({
        'building_id': building_ids.iloc[unlabeled_indices].values,
        'geo_level_1_id': geo_ids.values,
        'true_damage_grade': y_true.values if hasattr(y_true, 'values') else y_true,
        'predicted_damage_grade': y_pred,
        'correct_prediction': (y_true.values if hasattr(y_true, 'values') else y_true) == y_pred
    })
    
    predictions_path = f"{output_dir}/semi_supervised_predictions.csv"
    results_detailed.to_csv(predictions_path, index=False)
    print(f"Detailed predictions saved to: {predictions_path}")
    
    # Save geographic analysis
    geo_path = f"{output_dir}/semi_supervised_geo_analysis.csv"
    geo_results_df.to_csv(geo_path, index=False)
    print(f"Geographic analysis saved to: {geo_path}")
    
    # Save summary report
    summary_path = f"{output_dir}/semi_supervised_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SEMI-SUPERVISED LEARNING ROBUSTNESS CHECK - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: Nepal Earthquake Building Damage\n")
        f.write(f"Total samples: {len(building_ids)}\n")
        f.write(f"Labeled samples: {int(len(building_ids) * labeled_percentage)}\n")
        f.write(f"Unlabeled samples: {len(y_true)}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Algorithm: K-Nearest Neighbors (KNN)\n")
        f.write(f"  K value: {best_k}\n")
        f.write(f"  Distance metric: Euclidean\n")
        f.write(f"  Weighting: Distance-based\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Adjusted Rand Index (ARI): {metrics['ARI']:.4f}\n")
        f.write(f"  Normalized Mutual Information (NMI): {metrics['NMI']:.4f}\n")
        f.write(f"  Accuracy: {metrics['Accuracy']:.4f}\n\n")
        
        f.write("GEOGRAPHIC ANALYSIS:\n")
        f.write(f"  Number of regions: {len(geo_results_df)}\n")
        f.write(f"  Mean ARI across regions: {geo_results_df['ARI'].mean():.4f}\n")
        f.write(f"  Mean NMI across regions: {geo_results_df['NMI'].mean():.4f}\n")
        f.write(f"  Mean Accuracy across regions: {geo_results_df['Accuracy'].mean():.4f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("  - ARI measures clustering similarity (range: -1 to 1, higher is better)\n")
        f.write("  - NMI measures mutual information (range: 0 to 1, higher is better)\n")
        f.write("  - Both metrics evaluate how well pseudo-labels match true labels\n")
        f.write("  - Geographic variation indicates spatial heterogeneity in predictability\n")
    
    print(f"Summary report saved to: {summary_path}")
    print()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
def run_semi_supervised_analysis(labels_path, values_path, labeled_percentage=0.5,k_values=[3, 5, 7, 9, 11],output_dir='/home/claude'):
    print("\n" + "="*80)
    print("SEMI-SUPERVISED LEARNING ROBUSTNESS CHECK - MODULAR VERSION")
    print("="*80 + "\n")
    
    # Step 1: Load data
    df = load_and_prepare_data(labels_path, values_path)
    
    # Step 2: Preprocess
    X_scaled_df, y, building_ids, geo_level_1, label_encoders, scaler = preprocess_data(df)
    
    # Step 3: Split into labeled/unlabeled
    labeled_data, unlabeled_data = split_labeled_unlabeled(X_scaled_df, y, geo_level_1, labeled_percentage)
    
    # Step 4: Train KNN model
    knn_model, best_k, best_score = train_knn_model(labeled_data, k_values)
    
    # Step 5: Assign pseudo-labels
    y_unlabeled_pred = assign_pseudo_labels(knn_model, unlabeled_data)
    
    # Step 6: Evaluate distributions
    metrics = evaluate_label_distributions(unlabeled_data['y_true'], y_unlabeled_pred)
    
    # Step 7: Geographic analysis
    geo_results_df = analyze_geographic_subsets(
        unlabeled_data['y_true'], y_unlabeled_pred, unlabeled_data['geo']
    )
    
    # Step 8: Create visualizations
    viz_path = f"{output_dir}/semi_supervised_results.png"
    create_visualizations(
        unlabeled_data['y_true'], y_unlabeled_pred, 
        metrics, geo_results_df, viz_path
    )
    
    # Step 9: Save results
    save_detailed_results(
        building_ids, unlabeled_data['indices'],
        unlabeled_data['y_true'], y_unlabeled_pred,
        unlabeled_data['geo'], geo_results_df,
        metrics, best_k, labeled_percentage, output_dir
    )
    
    # Final summary
    print("="*80)
    print("SEMI-SUPERVISED LEARNING ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("KEY FINDINGS:")
    print(f"  1. Using {labeled_percentage*100:.0f}% labeled data, KNN achieved {metrics['Accuracy']:.1%} accuracy")
    print(f"  2. ARI = {metrics['ARI']:.4f}, NMI = {metrics['NMI']:.4f}")
    print(f"  3. Performance varies across {len(geo_results_df)} geographic regions")
    print(f"  4. Best K value: {best_k}")
    print()
    print("OUTPUT FILES:")
    print(f"  - {viz_path}")
    print(f"  - {output_dir}/semi_supervised_predictions.csv")
    print(f"  - {output_dir}/semi_supervised_geo_analysis.csv")
    print(f"  - {output_dir}/semi_supervised_summary.txt")
    print("="*80)
    
    # Return all results
    results = {
        'model': knn_model,
        'best_k': best_k,
        'metrics': metrics,
        'geo_results': geo_results_df,
        'predictions': y_unlabeled_pred,
        'true_labels': unlabeled_data['y_true'],
        'labeled_data': labeled_data,
        'unlabeled_data': unlabeled_data
    }
    
    return results


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================
if __name__ == "__main__":
    # Run the complete analysis
    results = run_semi_supervised_analysis(
        labels_path='./data/train_labels.csv',
        values_path='./data/train_values.csv',
        labeled_percentage=0.5,
        k_values=[3, 5, 7, 9, 11], #need to wait until model tunning 
        output_dir='./report'
    )
    
    print("\nAnalysis completed successfully!")
    print("You can now use the results for your proposal.")

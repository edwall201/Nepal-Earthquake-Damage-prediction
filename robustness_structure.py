# =============================================================================
# Robustness Check: Structural Feature Analysis
#
# Check 1 (Report Section 1.3): Foundation Type Subgroup Analysis
#   Trains a single model on the full training split, then evaluates weighted
#   F1 and severe damage recall separately on each foundation type subgroup
#   within the held-out test set only.
#
# Check 2 (Report Section 1.2): Superstructure Material Subgroup Analysis
#   Groups buildings into three seismic vulnerability tiers (Traditional,
#   Mixed/Transitional, Modern) and evaluates weighted F1 and severe damage
#   recall on the held-out test set only.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

RANDOM_SEED = 42

# ── Load data ──────────────────────────────────────────────────────────────────
train_values = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')
full_train_df = pd.merge(train_values, train_labels, on='building_id')

y = full_train_df['damage_grade'] - 1  # recode to 0/1/2


# ── Model builder ──────────────────────────────────────────────────────────────
def build_xgb():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ))
    ])


# =============================================================================
# CHECK 1: Foundation Type Subgroup Analysis (Report Section 1.3)
# =============================================================================

# ── Train on full training split ───────────────────────────────────────────────
X_encoded = pd.get_dummies(
    full_train_df.drop(columns=['building_id', 'damage_grade']),
    drop_first=False
)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

model = build_xgb()
model.fit(X_train, y_train)

overall_f1 = f1_score(y_test, model.predict(X_test), average='weighted')
overall_recall_severe = recall_score(
    y_test, model.predict(X_test), labels=[2], average='macro', zero_division=0
)
print(f"Overall F1 (test set):            {overall_f1:.4f}")
print(f"Overall Recall Severe (test set): {overall_recall_severe:.4f}\n")

# ── Evaluate by foundation_type on TEST SET only ───────────────────────────────
test_idx = X_test.index
foundation_types = full_train_df.loc[test_idx, 'foundation_type']

results = []
for ftype in sorted(foundation_types.unique()):
    mask = foundation_types == ftype
    X_sub = X_test[mask]
    y_sub = y_test[mask]

    if len(y_sub) < 50:
        print(f"foundation_type='{ftype}': too few samples ({len(y_sub)}), skipped")
        continue

    preds = model.predict(X_sub)
    f1 = f1_score(y_sub, preds, average='weighted')
    recall_severe = recall_score(y_sub, preds, labels=[2], average='macro', zero_division=0)
    damage_dist = y_sub.value_counts(normalize=True).sort_index().round(3).to_dict()

    results.append({
        'Foundation Type': ftype,
        'N (test)': len(y_sub),
        'Weighted F1': round(f1, 4),
        'Recall (Severe)': round(recall_severe, 4),
        'Damage Dist': damage_dist
    })
    print(f"foundation_type='{ftype}' | n={len(y_sub):5d} | "
          f"F1={f1:.4f} | Recall(severe)={recall_severe:.4f} | "
          f"damage dist={damage_dist}")

foundation_results_df = pd.DataFrame(results)
print("\nSummary Table:")
print(foundation_results_df[['Foundation Type', 'N (test)', 'Weighted F1', 'Recall (Severe)']].to_string(index=False))

# ── Plot: Foundation type F1 and Recall side by side ──────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# F1
colors_f1 = ['#d7191c' if f < overall_f1 - 0.05 else '#2c7bb6'
              for f in foundation_results_df['Weighted F1']]
bars1 = ax1.bar(foundation_results_df['Foundation Type'], foundation_results_df['Weighted F1'],
                color=colors_f1, edgecolor='white', width=0.5, zorder=3)
for bar, score in zip(bars1, foundation_results_df['Weighted F1']):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.axhline(y=overall_f1, color='#2c7bb6', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Foundation Type', fontsize=12)
ax1.set_ylabel('Weighted F1-Score', fontsize=12)
ax1.set_ylim(foundation_results_df['Weighted F1'].min() - 0.05,
             foundation_results_df['Weighted F1'].max() + 0.05)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
legend_f1 = [
    mpatches.Patch(facecolor='#2c7bb6', label='Within 5pp of overall F1'),
    mpatches.Patch(facecolor='#d7191c', label='>5pp below overall F1'),
    plt.Line2D([0], [0], color='#2c7bb6', linestyle='--',
               linewidth=1.5, label=f'Overall F1 = {overall_f1:.3f}')
]
ax1.legend(handles=legend_f1, fontsize=9, loc='lower right')

# Recall
colors_r = ['#d7191c' if r < overall_recall_severe - 0.05 else '#2c7bb6'
            for r in foundation_results_df['Recall (Severe)']]
bars2 = ax2.bar(foundation_results_df['Foundation Type'], foundation_results_df['Recall (Severe)'],
                color=colors_r, edgecolor='white', width=0.5, zorder=3)
for bar, score in zip(bars2, foundation_results_df['Recall (Severe)']):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.axhline(y=overall_recall_severe, color='#2c7bb6', linestyle='--', linewidth=1.5, alpha=0.8)
ax2.set_xlabel('Foundation Type', fontsize=12)
ax2.set_ylabel('Recall -- Severe Damage (Grade 3)', fontsize=12)
ax2.set_ylim(max(0, foundation_results_df['Recall (Severe)'].min() - 0.08),
             min(1, foundation_results_df['Recall (Severe)'].max() + 0.05))
ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
legend_r = [
    mpatches.Patch(facecolor='#2c7bb6', label='Within 5pp of overall recall'),
    mpatches.Patch(facecolor='#d7191c', label='>5pp below overall recall'),
    plt.Line2D([0], [0], color='#2c7bb6', linestyle='--',
               linewidth=1.5, label=f'Overall Recall = {overall_recall_severe:.3f}')
]
ax2.legend(handles=legend_r, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('foundation_combined.png', dpi=200, bbox_inches='tight')
plt.show()


# =============================================================================
# CHECK 2: Superstructure Material Subgroup Analysis (Report Section 1.2)
# =============================================================================

# ── Define material groups ─────────────────────────────────────────────────────
# Priority: Traditional > Mixed > Modern
# A building with mixed materials is assigned to the most fragile category.
def assign_material_group(row):
    if (row['has_superstructure_mud_mortar_stone'] or
            row['has_superstructure_adobe_mud'] or
            row['has_superstructure_stone_flag']):
        return 'Traditional\n(Mud/Stone)'
    elif (row['has_superstructure_cement_mortar_brick'] or
              row['has_superstructure_timber'] or
              row['has_superstructure_mud_mortar_brick'] or
              row['has_superstructure_bamboo']):
        return 'Mixed/Transitional\n(Brick/Timber)'
    elif (row['has_superstructure_rc_engineered'] or
              row['has_superstructure_rc_non_engineered'] or
              row['has_superstructure_cement_mortar_stone']):
        return 'Modern\n(RC/Cement)'
    else:
        return 'Other'


full_train_df['material_group'] = full_train_df.apply(assign_material_group, axis=1)

print("\nMaterial Group Distribution:")
print(full_train_df['material_group'].value_counts())
print()
for grp in full_train_df['material_group'].unique():
    subset = full_train_df[full_train_df['material_group'] == grp]
    dmg = subset['damage_grade'].value_counts(normalize=True).sort_index().round(3)
    print(f"{grp.replace(chr(10), ' ')} (n={len(subset):,}): damage dist = {dmg.to_dict()}")

# ── Train model (drop material_group from features) ───────────────────────────
X_encoded_mat = pd.get_dummies(
    full_train_df.drop(columns=['building_id', 'damage_grade', 'material_group']),
    drop_first=False
)

X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(
    X_encoded_mat, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

model_mat = build_xgb()
model_mat.fit(X_train_mat, y_train_mat)

overall_f1_mat = f1_score(y_test_mat, model_mat.predict(X_test_mat), average='weighted')
overall_recall_mat = recall_score(
    y_test_mat, model_mat.predict(X_test_mat), labels=[2], average='macro', zero_division=0
)
print(f"\nOverall F1 (test set):            {overall_f1_mat:.4f}")
print(f"Overall Recall Severe (test set): {overall_recall_mat:.4f}\n")

# ── Evaluate on TEST SET only, grouped by material type ───────────────────────
test_idx_mat = X_test_mat.index
material_groups = full_train_df.loc[test_idx_mat, 'material_group']

GROUP_ORDER = [
    'Traditional\n(Mud/Stone)',
    'Mixed/Transitional\n(Brick/Timber)',
    'Modern\n(RC/Cement)'
]

mat_results = []
for grp in GROUP_ORDER:
    mask = material_groups == grp
    X_sub = X_test_mat[mask]
    y_sub = y_test_mat[mask]

    if len(y_sub) < 50:
        print(f"'{grp}': too few samples ({len(y_sub)}), skipped")
        continue

    preds = model_mat.predict(X_sub)
    f1 = f1_score(y_sub, preds, average='weighted')
    recall_severe = recall_score(y_sub, preds, labels=[2], average='macro', zero_division=0)
    damage_dist = y_sub.value_counts(normalize=True).sort_index().round(3).to_dict()

    mat_results.append({
        'Group': grp,
        'N (test)': len(y_sub),
        'Weighted F1': round(f1, 4),
        'Recall (Severe)': round(recall_severe, 4),
        'Grade3 %': round(damage_dist.get(2, 0) * 100, 1)
    })
    print(f"{grp.replace(chr(10), ' '):<35} | n={len(y_sub):6,} | "
          f"F1={f1:.4f} | Recall(severe)={recall_severe:.4f} | "
          f"grade3={damage_dist.get(2, 0) * 100:.1f}%")

mat_results_df = pd.DataFrame(mat_results)
print("\n=== Summary Table ===")
print(mat_results_df[['Group', 'N (test)', 'Grade3 %', 'Weighted F1', 'Recall (Severe)']].to_string(index=False))


# ── Plot helper ────────────────────────────────────────────────────────────────
def make_bar_chart(ax, groups, values, baseline, ylabel,
                   threshold=0.05, color_above='#2c7bb6', color_below='#d7191c'):
    colors = [color_below if v < baseline - threshold else color_above for v in values]
    bars = ax.bar(groups, values, color=colors, edgecolor='white', width=0.45, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(y=baseline, color='#2c7bb6', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(max(0, min(values) - 0.12), min(1.0, max(values) + 0.08))
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_elements = [
        mpatches.Patch(facecolor=color_above, label='Within 5pp of overall'),
        mpatches.Patch(facecolor=color_below, label='>5pp below overall'),
        plt.Line2D([0], [0], color='#2c7bb6', linestyle='--',
                   linewidth=1.5, label=f'Overall = {baseline:.3f}')
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')


# ── Plot: Material F1 and Recall side by side ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

make_bar_chart(ax1, mat_results_df['Group'], mat_results_df['Weighted F1'],
               baseline=overall_f1_mat, ylabel='Weighted F1-Score')
fig.text(0.27, -0.02,
         'XGBoost Performance by Superstructure Material Group (Weighted F1, Test Set Only)',
         ha='center', fontsize=9, style='italic')

make_bar_chart(ax2, mat_results_df['Group'], mat_results_df['Recall (Severe)'],
               baseline=overall_recall_mat, ylabel='Recall -- Severe Damage (Grade 3)')
fig.text(0.75, -0.02,
         'Risk of Missing Severely Damaged Buildings by Material Group (Test Set Only)',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('material_combined.png', dpi=200, bbox_inches='tight')
plt.show()

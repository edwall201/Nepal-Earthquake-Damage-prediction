# =============================================================================
# Robustness Check: Geographic Analysis
#
# Check 1 (Appendix): Geographic Feature Ablation
#   Trains separate models while systematically removing geo features at each
#   level of granularity. Quantifies the contribution of geographic data to
#   predictive performance.
#
# Check 2 (Report Section 1.1): District-Level Subgroup Analysis
#   Trains a single model on the full training split, then evaluates performance
#   separately on each geo_level_1 (district) subgroup within the held-out test
#   set only. Ensures no data leakage and reflects genuine generalization behavior.
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_SEED = 42

# ── Load data ──────────────────────────────────────────────────────────────────
train_values = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')
full_train_df = pd.merge(train_values, train_labels, on='building_id')

y = full_train_df['damage_grade'] - 1  # recode to 0/1/2

print("Dataset shape:", full_train_df.shape)
print("Damage grade distribution:")
print(full_train_df['damage_grade'].value_counts(normalize=True).sort_index().round(4))


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
# CHECK 1: Geographic Feature Ablation (Appendix)
# =============================================================================

def run_model(X, y):
    X = pd.get_dummies(X, drop_first=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model = build_xgb()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds, average='weighted')


results = {}

X_full = full_train_df.drop(columns=['building_id', 'damage_grade'])
results['Full Model\n(All Geo)'] = run_model(X_full, y)

X_geo1 = full_train_df.drop(columns=['building_id', 'damage_grade', 'geo_level_2_id', 'geo_level_3_id'])
results['District\nLevel Only'] = run_model(X_geo1, y)

X_geo2 = full_train_df.drop(columns=['building_id', 'damage_grade', 'geo_level_1_id', 'geo_level_3_id'])
results['Sub-district\nLevel Only'] = run_model(X_geo2, y)

X_geo3 = full_train_df.drop(columns=['building_id', 'damage_grade', 'geo_level_1_id', 'geo_level_2_id'])
results['Ward\nLevel Only'] = run_model(X_geo3, y)

X_no_geo = full_train_df.drop(columns=['building_id', 'damage_grade', 'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'])
results['No Geo\nFeatures'] = run_model(X_no_geo, y)

print("\nGeo Ablation Results:")
for k, v in results.items():
    print(f"  {k.replace(chr(10), ' ')}: {v:.4f}")

# ── Plot: Geo Ablation ─────────────────────────────────────────────────────────
labels = list(results.keys())
f1_scores = list(results.values())
colors = ['#2c7bb6'] + ['#abd9e9'] * 3 + ['#d7191c']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, f1_scores, color=colors, edgecolor='white', width=0.55, zorder=3)

for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

baseline = results['Full Model\n(All Geo)']
ax.set_ylim(0.54, 0.76)
ax.set_ylabel('Weighted F1-Score', fontsize=12)
ax.axhline(y=baseline, color='#2c7bb6', linestyle='--', linewidth=1.2, alpha=0.6, zorder=2)
ax.text(len(labels) - 0.55, baseline + 0.002, 'Baseline', color='#2c7bb6', fontsize=9)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [
    Patch(facecolor='#2c7bb6', label='Full Model (Baseline)'),
    Patch(facecolor='#abd9e9', label='Partial Geographic Data'),
    Patch(facecolor='#d7191c', label='No Geographic Features')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.8)

fig.text(0.5, -0.05,
         'Impact of Geographic Feature Removal on Model Performance (XGBoost, Weighted F1-Score)',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('geo_ablation_xgb.png', dpi=200, bbox_inches='tight')
plt.show()


# =============================================================================
# CHECK 2: District-Level Subgroup Analysis (Report Section 1.1)
# =============================================================================

# ── Train on full training split ───────────────────────────────────────────────
X_encoded = pd.get_dummies(
    full_train_df.drop(columns=['building_id', 'damage_grade']), drop_first=False
)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

model_full = build_xgb()
model_full.fit(X_train_full, y_train_full)

overall_f1 = f1_score(y_test_full, model_full.predict(X_test_full), average='weighted')
print(f"\nOverall F1 (test set): {overall_f1:.4f}")

# ── Evaluate per district on TEST SET only ─────────────────────────────────────
test_idx = X_test_full.index
geo_test = full_train_df.loc[test_idx, 'geo_level_1_id']

geo_results = []
for geo_id in sorted(geo_test.unique()):
    mask = geo_test == geo_id
    X_sub = X_test_full[mask]
    y_sub = y_test_full[mask]

    if len(y_sub) < 50:
        print(f"District {geo_id}: too few samples ({len(y_sub)}), skipped")
        continue

    preds = model_full.predict(X_sub)
    f1 = f1_score(y_sub, preds, average='weighted')
    geo_results.append({
        'District ID': geo_id,
        'Sample Size': len(y_sub),
        'Weighted F1': round(f1, 4)
    })

geo_df = pd.DataFrame(geo_results).sort_values('Weighted F1', ascending=False)

print(f"Number of districts evaluated: {len(geo_df)}")
print(f"F1 range: {geo_df['Weighted F1'].min():.4f} - {geo_df['Weighted F1'].max():.4f}")
print(f"F1 mean:  {geo_df['Weighted F1'].mean():.4f}")
print(f"F1 std:   {geo_df['Weighted F1'].std():.4f}")
print(f"Districts >5pp below baseline: {(geo_df['Weighted F1'] < overall_f1 - 0.05).sum()}")
print("\nFull Table:")
print(geo_df.to_string(index=False))

# ── Plot: District-level performance ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

colors_geo = ['#d7191c' if f < overall_f1 - 0.05 else '#2c7bb6'
              for f in geo_df['Weighted F1']]

ax.bar(geo_df['District ID'].astype(str), geo_df['Weighted F1'],
       color=colors_geo, edgecolor='white', width=0.7, zorder=3)

ax.axhline(y=overall_f1, color='#2c7bb6', linestyle='--', linewidth=1.5,
           alpha=0.8, zorder=2)

ax.set_xlabel('District ID (geo_level_1)', fontsize=11)
ax.set_ylabel('Weighted F1-Score', fontsize=11)
ax.set_ylim(geo_df['Weighted F1'].min() - 0.05, geo_df['Weighted F1'].max() + 0.05)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', rotation=45)

legend_elements = [
    Patch(facecolor='#2c7bb6', label='Within 5pp of overall F1'),
    Patch(facecolor='#d7191c', label='>5pp below overall F1'),
    plt.Line2D([0], [0], color='#2c7bb6', linestyle='--',
               linewidth=1.5, label=f'Overall F1 = {overall_f1:.3f}')
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower left')

plt.tight_layout()
fig.text(0.5, -0.05,
         'XGBoost Performance Across Districts (geo_level_1 subgroups, Weighted F1-Score)',
         ha='center', fontsize=9, style='italic')
plt.savefig('geo_level1_subgroup.png', dpi=200, bbox_inches='tight')
plt.show()

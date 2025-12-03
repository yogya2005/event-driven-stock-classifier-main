"""
Overfitting Analysis for Stock Classifier Models

This script tests for overfitting by:
1. Comparing train vs test accuracy (gap indicates overfitting)
2. Performing k-fold cross-validation
3. Creating learning curves
4. Analyzing per-class performance differences
"""

import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

print("=" * 80)
print("OVERFITTING ANALYSIS")
print("=" * 80)

# Load train/test split
print("\nLoading data...")
data = np.load('../data/train_test_split_all.npz', allow_pickle=True)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Helper to unwrap and ensure CSR sparse matrices
import numpy as _np
from scipy import sparse as _sparse

def _ensure_csr(X):
    # Unwrap object arrays that contain a single sparse matrix
    if isinstance(X, _np.ndarray) and X.dtype == object:
        if X.shape == ():
            X = X.item()
        elif X.shape == (1,):
            X = X[0]
    # If already sparse, return CSR
    if _sparse.issparse(X):
        return X.tocsr()
    # Otherwise convert dense to CSR
    return _sparse.csr_matrix(X)

X_train = _ensure_csr(X_train)
X_test = _ensure_csr(X_test)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Load models
print("\nLoading models...")
with open('../models/logistic_regression_model_all.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('../models/random_forest_model_all.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('../models/xgboost_model_all.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# ============================================================================
# TEST 1: Train vs Test Accuracy Gap
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Train vs Test Accuracy Gap")
print("=" * 80)
print("\nA large gap (>5-10%) suggests overfitting.\n")

results = {}
for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Train accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Test accuracy
    y_test_pred = model.predict(X_test)
    # Handle XGBoost label offset
    if name == 'XGBoost':
        y_test_pred = y_test_pred - 1
    test_acc = accuracy_score(y_test, y_test_pred)
    
    gap = train_acc - test_acc
    gap_pct = (gap / train_acc) * 100
    
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Gap:            {gap:.4f} ({gap*100:.2f} pp)")
    print(f"  Relative Gap:   {gap_pct:.2f}%")
    
    if gap_pct < 5:
        print(f"  Good generalization (gap < 5%)")
    elif gap_pct < 10:
        print(f"  Slight overfitting (5% < gap < 10%)")
    else:
        print(f"  Significant overfitting (gap > 10%)")
    
    results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'gap_pct': gap_pct
    }

# ============================================================================
# TEST 2: Cross-Validation Scores
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Cross-Validation Analysis (3-fold, optional subsample)")
print("=" * 80)
print("\nHigh variance in CV scores suggests overfitting.\n")

cv_results = {}
from sklearn.utils import check_random_state

# Optional speed-up: subsample for CV if extremely large
max_cv_rows = 200_000
use_X = X_train
use_y = y_train
if X_train.shape[0] > max_cv_rows:
    rng = check_random_state(42)
    idx = rng.choice(X_train.shape[0], size=max_cv_rows, replace=False)
    use_X = X_train[idx]
    use_y = y_train[idx]
    print(f"Subsampled training set for CV: {use_X.shape[0]:,} rows")

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Perform 3-fold cross-validation on (possibly subsampled) training data
    # Note: For XGBoost, need to adjust labels
    if name == 'XGBoost':
        y_train_cv = use_y + 1
    else:
        y_train_cv = use_y
    
    cv_scores = cross_val_score(model, use_X, y_train_cv, cv=3, 
                                 scoring='accuracy', n_jobs=-1)
    
    print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean:      {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"  Std Dev:   {cv_scores.std():.4f} ({cv_scores.std()*100:.2f} pp)")
    print(f"  Min-Max:   {cv_scores.min():.4f} - {cv_scores.max():.4f}")
    
    variance = cv_scores.std() * 100
    if variance < 1:
        print(f"  Low variance (std < 1%)")
    elif variance < 2:
        print(f"  Moderate variance (1% < std < 2%)")
    else:
        print(f"  High variance (std > 2%)")
    
    cv_results[name] = {
        'scores': cv_scores,
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }

# ============================================================================
# TEST 3: Per-Class Performance Analysis
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Per-Class Performance (Train vs Test)")
print("=" * 80)
print("\nLarge differences in per-class accuracy suggest overfitting.\n")

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Train predictions
    y_train_pred = model.predict(X_train)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    if name == 'XGBoost':
        y_test_pred = y_test_pred - 1
    
    # Calculate per-class accuracy
    for label, label_name in [(-1, 'Negative'), (0, 'Neutral'), (1, 'Positive')]:
        # Train accuracy for this class
        train_mask = y_train == label
        if train_mask.sum() > 0:
            train_class_acc = (y_train_pred[train_mask] == label).sum() / train_mask.sum()
        else:
            train_class_acc = 0
        
        # Test accuracy for this class
        test_mask = y_test == label
        if test_mask.sum() > 0:
            test_class_acc = (y_test_pred[test_mask] == label).sum() / test_mask.sum()
        else:
            test_class_acc = 0
        
        class_gap = (train_class_acc - test_class_acc) * 100
        print(f"  {label_name:8s}: Train={train_class_acc*100:5.2f}%  Test={test_class_acc*100:5.2f}%  Gap={class_gap:+6.2f}pp")

# ============================================================================
# TEST 4: Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

# Create comprehensive overfitting visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Train vs Test Accuracy
ax1 = axes[0, 0]
model_names = list(results.keys())
train_accs = [results[m]['train_acc'] * 100 for m in model_names]
test_accs = [results[m]['test_acc'] * 100 for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width/2, train_accs, width, label='Train', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_accs, width, label='Test', color='#3498db', alpha=0.8)

ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Train vs Test Accuracy\n(Small gap = good generalization)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=33.33, color='red', linestyle='--', alpha=0.5, label='Random')

# Add gap annotations
for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    gap = train - test
    ax1.text(i, max(train, test) + 1, f'Δ{gap:.1f}pp', 
             ha='center', fontsize=9, fontweight='bold')

# Plot 2: Accuracy Gap
ax2 = axes[0, 1]
gaps = [results[m]['gap'] * 100 for m in model_names]
colors = ['green' if g < 5 else 'orange' if g < 10 else 'red' for g in gaps]
bars = ax2.bar(model_names, gaps, color=colors, alpha=0.7)
ax2.set_ylabel('Train-Test Gap (pp)', fontsize=11)
ax2.set_title('Overfitting Gap\n(Green=Good, Orange=Moderate, Red=High)', fontsize=12, fontweight='bold')
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Add value labels
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{gap:.2f}pp', ha='center', va='bottom', fontsize=10)

# Plot 3: Cross-Validation Scores Distribution
ax3 = axes[1, 0]
positions = []
cv_data = []
for i, name in enumerate(model_names):
    scores = cv_results[name]['scores'] * 100
    positions.extend([i] * len(scores))
    cv_data.extend(scores)

scatter = ax3.scatter(positions, cv_data, alpha=0.6, s=100, c=positions, cmap='viridis')
for i, name in enumerate(model_names):
    mean = cv_results[name]['mean'] * 100
    ax3.plot([i-0.2, i+0.2], [mean, mean], 'r-', linewidth=2)

ax3.set_ylabel('CV Accuracy (%)', fontsize=11)
ax3.set_title('Cross-Validation Score Distribution\n(Red line = mean, dots = individual folds)', 
              fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels(model_names, rotation=15, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Summary Table
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary table
table_data = []
table_data.append(['Model', 'Train Acc', 'Test Acc', 'Gap', 'CV Mean±Std', 'Status'])
for name in model_names:
    train = f"{results[name]['train_acc']*100:.2f}%"
    test = f"{results[name]['test_acc']*100:.2f}%"
    gap = f"{results[name]['gap']*100:.2f}pp"
    cv = f"{cv_results[name]['mean']*100:.2f}±{cv_results[name]['std']*100:.2f}%"
    
    if results[name]['gap_pct'] < 5:
        status = 'Good'
    elif results[name]['gap_pct'] < 10:
        status = 'OK'
    else:
        status = 'Overfit'
    
    table_data.append([name.replace(' ', '\n'), train, test, gap, cv, status])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.12, 0.12, 0.12, 0.18, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Summary Table', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../visualizations/overfitting_analysis_all.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: overfitting_analysis_all.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("\nOverall Assessment:")
for name in model_names:
    gap = results[name]['gap_pct']
    cv_std = cv_results[name]['std'] * 100
    
    print(f"\n{name}:")
    if gap < 5 and cv_std < 2:
        print("  EXCELLENT - Good generalization, low variance")
        print("  → Model is NOT overfitting")
    elif gap < 10 and cv_std < 3:
        print("  ACCEPTABLE - Slight overfitting but manageable")
        print("  → Model generalizes reasonably well")
    else:
        print("  CONCERNING - Significant overfitting detected")
        print("  → Consider regularization or simpler model")

print("\n" + "=" * 80)
print("\nIf overfitting detected, try these fixes:")
print("  1. Increase regularization (C parameter for LR, max_depth for trees)")
print("  2. Reduce model complexity (fewer features, simpler architecture)")
print("  3. Add more training data (already using full dataset)")
print("  4. Use ensemble methods with stronger regularization")
print("  5. Feature selection to remove noisy features")

print("\nOverfitting analysis complete!")
print("  Check overfitting_analysis_all.png for visualizations")

"""
Check current model predictions to diagnose class imbalance issue
"""
import pickle
import numpy as np

# Load test data
data = np.load('../data/train_test_split.npz', allow_pickle=True)
X_test = data['X_test']
y_test = data['y_test']

# Convert from object array to sparse matrix if needed
if isinstance(X_test, np.ndarray) and X_test.dtype == object:
    X_test = X_test.item()

print("=" * 70)
print("ACTUAL TEST SET DISTRIBUTION")
print("=" * 70)
unique, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique, counts):
    pct = count / len(y_test) * 100
    print(f"Class {label:2d}: {count:5d} samples ({pct:5.2f}%)")

# Check each model's predictions
models = {
    'Logistic Regression': '../models/logistic_regression_model.pkl',
    'Random Forest': '../models/random_forest_model.pkl',
    'XGBoost': '../models/xgboost_model.pkl'
}

for model_name, model_file in models.items():
    print(f"\n{'=' * 70}")
    print(f"{model_name.upper()} PREDICTIONS")
    print("=" * 70)
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    if model_name == 'XGBoost':
        # XGBoost uses 0, 1, 2 labels, need to convert back
        y_pred = model.predict(X_test) - 1
    else:
        y_pred = model.predict(X_test)
    
    # Show prediction distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique_pred, counts_pred):
        pct = count / len(y_pred) * 100
        print(f"Predicted {label:2d}: {count:5d} samples ({pct:5.2f}%)")
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

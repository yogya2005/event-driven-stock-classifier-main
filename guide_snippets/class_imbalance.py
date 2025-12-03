from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# 1. Logistic Regression with class_weight='balanced'
lr_model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    class_weight='balanced'  # Automatically adjust weights inversely proportional to class frequencies
)

# 2. Random Forest with class_weight='balanced'
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced'  # Handle imbalance in tree construction
)

# 3. XGBoost with sample weights
# Calculate weights for each sample in the training set
sample_weights = compute_sample_weight('balanced', y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    objective='multi:softmax',
    num_class=3
)
# Pass sample_weights during training
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

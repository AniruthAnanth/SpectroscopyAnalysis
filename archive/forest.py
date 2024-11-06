import numpy as np
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

# Load data
with open('data.json', 'r') as f:
    raw_data = json.load(f)

def convert_data_point(data_point):
    wl = np.array(data_point["wl"])
    r = np.array(data_point["r"])
    c = np.array([item[2] for item in data_point["l"]])
    return wl, r, c

X, y, wl = [], [], None

for data_point in raw_data:
    wl_, r, c = convert_data_point(data_point)
    X.append(r)  # Assuming `r` is already a 1D array
    y.append(c)
    wl = wl_

# Normalize the data
X = preprocessing.normalize(np.array(X))
y = preprocessing.normalize(np.array(y))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize model for hyperparameter search
random_forest_model = RandomForestRegressor(random_state=42, verbose=1)

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=random_forest_model,
    param_distributions=param_dist,
    n_iter=50,  # Number of different combinations to try
    cv=5,       # 5-Fold Cross-Validation
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)

# Fit random search
print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Output the best parameters
print(f"Best Parameters: {random_search.best_params_}")

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature Importances:")
for i in indices:
    print(f"Feature {i}: {importances[i]:.6f}")

# Save the model if needed
joblib.dump(best_model, 'random_forest_best_model.pkl')
print("Model saved as 'random_forest_best_model.pkl'")


# Evaluate cross-validation performance
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"Mean CV MAE: {-cv_scores.mean():.6f}")

# Train the best model on full training data
print("Training best model on full training data...")
best_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)

print(f"Test Mean Absolute Error (MAE): {mae:.6f}")
#print(f"Test Root Mean Squared Error (RMSE): {rmse:.6f}")

# Feature Importance Analysis

# Setup and Imports
import pandas as pd
import numpy as np
import warnings
import pickle
import hashlib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from foundry.transforms import Dataset

# Configuration
warnings.filterwarnings('ignore')
sns.set(style='whitegrid', rc={'figure.figsize':(12,8)})

# Data Loading and Initial Cleaning

# Load the dataset
df_raw = Dataset.get("po_header_and_item_restricted_24_25").read_table(format="pandas")

# Initial cleaning and type correction
df = df_raw.copy()
numeric_cols_to_check = ['item_net_price', 'item_purchase_order_quantity', 'net_order_value', 'item_planned_delivery_time_in_days']
for col in numeric_cols_to_check:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where the target and the supplier is null, as they are unusable for training
df.dropna(subset=['item_planned_delivery_time_in_days', 'vendor_name'], inplace=True)
print(f"  > Initial data cleaned. Shape is now: {df.shape}")


# Data Preparation

# Prune irrelevant and low-value columns 
cols_to_drop_identifiers = [col for col in df.columns if '_id' in col or '_number' in col]
cols_to_drop_identifiers.append('internal_reference')
df.drop(columns=list(set(cols_to_drop_identifiers)), inplace=True, errors='ignore')

high_missing_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.90]
df.drop(columns=high_missing_cols, inplace=True, errors='ignore')

low_variance_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
df.drop(columns=low_variance_cols, inplace=True, errors='ignore')
print(f"  > Pruning complete. Shape is now: {df.shape}")

# Define Feature and Target Sets 
target = 'item_planned_delivery_time_in_days'
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]

# Create Train, Validation, and Test splits
# This three-way split ensures a robust evaluation process.
# First, split into a training+validation set (80%) and a final test set (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Then, split the 80% into a training set (80% of 80% = 64%) and a validation set (20% of 80% = 16%)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
print(f"  > Data split into {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples.")

# Anonymize vendor name using a one-way hash
if 'vendor_name' in X_train.columns:
    for dataset in [X_train, X_val, X_test, X_train_full]:
        dataset['vendor_anonymous_id'] = dataset['vendor_name'].apply(
            lambda x: hashlib.sha256(str(x).encode('utf-8')).hexdigest()
        )
        dataset.drop(columns=['vendor_name'], inplace=True)

# Model Comparison on Validation Set

# Define preprocessing steps for numerical and categorical features
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
if 'vendor_anonymous_id' in numerical_features:
    numerical_features.remove('vendor_anonymous_id')
    categorical_features.append('vendor_anonymous_id')

# Create a preprocessor to handle different feature types
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), numerical_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), categorical_features)
], remainder='drop')

# Define the regression models to be compared
regressors = {
    "Dummy Regressor (Baseline)": DummyRegressor(strategy='mean'),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=13),
    "Random Forest": RandomForestRegressor(random_state=67, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=78)
}

# Build a full pipeline for each model
pipelines = {name: Pipeline([('preprocessor', preprocessor), ('regressor', regressor)]) for name, regressor in regressors.items()}

# Evaluate each model on the validation set
evaluation_results = {}
for name, pipeline in pipelines.items():
    print(f"\n  > Evaluating: {name}")
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    medae = median_absolute_error(y_val, y_val_pred)
    evaluation_results[name] = {'R-squared': r2, 'MAE': mae, 'RMSE': rmse, 'MedAE': medae}
    print(f"    - R-squared: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MedAE: {medae:.2f}")


# Display the comparative results
results_df = pd.DataFrame(evaluation_results).T
results_df = results_df.sort_values(by='R-squared', ascending=False)
print("\nModel Performance on Validation Set:")
display(results_df)

# Hyperparameter Tuning


best_model_name = results_df.index[0]
print(f"  > Top performing model is '{best_model_name}'. Proceeding to tune it.")

# Define parameter grids for tunable models
param_grids = {
    "Random Forest": {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [10, 20, 30],
        'regressor__min_samples_leaf': [1, 2, 4]
    },
    "Gradient Boosting": {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5, 7]
    }
}

param_grid_for_best_model = param_grids.get(best_model_name, None)

if param_grid_for_best_model is None:
    print(f"  > No parameter grid defined for '{best_model_name}'. Using baseline model as final.")
    tuned_model = pipelines[best_model_name]
else:
    random_search = RandomizedSearchCV(
        pipelines[best_model_name], param_distributions=param_grid_for_best_model, n_iter=10,
        cv=5, scoring='r2', n_jobs=-1, random_state=59
    )
    print(f"  > Running RandomizedSearchCV for {best_model_name}...")
    # Tuning is performed on the training set
    random_search.fit(X_train, y_train)
    print(f"  > Best parameters found: {random_search.best_params_}")
    tuned_model = random_search.best_estimator_

# Final Evaluation and Interpretation

# Final Model Training ---
# Retrain the final model on the combined training and validation data
final_model = tuned_model
final_model.fit(X_train_full, y_train_full)

# Final Evaluation on Holdout Test Set 
print("\n  > Evaluating final model on unseen test set...")
y_pred_test = final_model.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
medae_test = median_absolute_error(y_test, y_pred_test)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out cases where the true value is zero to avoid division by zero
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0 # Return 0 if all true values are zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

mape_test = calculate_mape(y_test, y_pred_test)

print(f"\nFinal Model Performance on Holdout Test Set:")
print(f"  - R-squared (R²): {r2_test:.3f}")
print(f"  - Mean Absolute Error (MAE): {mae_test:,.2f} days")
print(f"  - Median Absolute Error (MedAE): {medae_test:,.2f} days")
print(f"  - Root Mean Squared Error (RMSE): {rmse_test:,.2f} days")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape_test:.2f}%")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.xlabel("Actual Planned Lead Time (Days)")
plt.ylabel("Predicted Planned Lead Time (Days)")
plt.title(f"Final Model: Actual vs. Predicted Lead Time for {best_model_name}")
plt.show()

# Feature Importance 
print("\n  > Calculating feature importances...")
perm_importance = permutation_importance(
    final_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='r2'
)
sorted_idx = perm_importance.importances_mean.argsort()
importance_df = pd.DataFrame(
    perm_importance.importances[sorted_idx].T,
    columns=X_test.columns[sorted_idx],
)
fig, ax = plt.subplots(figsize=(12, 10))
importance_df.plot.box(vert=False, whis=10, ax=ax)
ax.set_title(f"Permutation Importances for {best_model_name} (Test Set)")
ax.set_xlabel("Decrease in R-squared Score")
plt.tight_layout()
plt.show()

# Save Final Model 
model_filename = 'final_lead_time_regressor.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(final_model, f)
print(f"\nFinal, tuned model pipeline saved to '{model_filename}'")
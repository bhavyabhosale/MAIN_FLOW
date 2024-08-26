# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('/mnt/data/heartmf.csv')

# Explore the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nData summary:")
print(data.describe())
print("\nMissing values:")
print(data.isnull().sum())

# Feature Engineering
# Example: Creating interaction features, polynomial features, etc.
# Here, we'll create a new feature as a simple example.
data['age_slope_interaction'] = data['age'] * data['slope']

# Split the data into features and target
X = data.drop('target', axis=1)  # Assuming 'target' is the name of the target column
y = data['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection using PCA
pca = PCA(n_components=10)  # Example: Reduce to 10 principal components
X_pca = pca.fit_transform(X_scaled)

# Feature Selection using Feature Importance (Random Forest)
model_rf = RandomForestClassifier()
model_rf.fit(X_scaled, y)

# Get feature importance
importances = model_rf.feature_importances_
indices = importances.argsort()[::-1]

# Print feature ranking
print("\nFeature ranking:")
for i in range(X.shape[1]):
    print(f"{i + 1}. Feature {X.columns[indices[i]]} ({importances[indices[i]]})")

# Select the top N features (e.g., top 10 features)
top_n_features = 10
selected_features = X.columns[indices[:top_n_features]]
X_selected = X[selected_features]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Model Training using XGBoost
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

# Model Evaluation
y_pred = model_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel accuracy:", accuracy)

# Cross-validation for robust performance evaluation
cv_scores = cross_val_score(model_xgb, X_selected, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

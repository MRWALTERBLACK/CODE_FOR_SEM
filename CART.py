# CART - Insurance Charges Prediction


# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
# Replace the path with your actual dataset location
df = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\insurance.csv")

# Step 3: Encode Categorical Variables
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])        # male=1, female=0
df['smoker'] = le.fit_transform(df['smoker'])  # yes=1, no=0
df['region'] = le.fit_transform(df['region'])  # northeast=0, northwest=1, southeast=2, southwest=3

# Step 4: Define Features (X) and Target (y)
X = df.drop('charges', axis=1)
y = df['charges']

# Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train CART Model (Decision Tree Regressor)
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.title("CART - Regression Tree for Insurance Charges")
plt.show()

# Step 10: Plot Feature Importances
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=model.feature_importances_)
plt.title("Feature Importances in Insurance Charge Prediction")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

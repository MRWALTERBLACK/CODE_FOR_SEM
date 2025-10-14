
# **Step 1: IMPORT LIBRARIES**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split

# **Step 2: LOAD DATASET**
df = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\insurance.csv")

# **Step 3: DEFINE PREDICTORS AND OUTCOME**
predictors = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
outcome = 'charges'

# **Step 4: ONE-HOT ENCODE CATEGORICAL VARIABLES**
df_encoded = pd.get_dummies(df[predictors], drop_first=True).astype(float)

# **Step 5: FIT MODEL USING SKLEARN**
lr = LinearRegression()
lr.fit(df_encoded, df[outcome])

# **Step 6: PRINT COEFFICIENTS**
print(f'Intercept: {lr.intercept_:.2f}')
print('Coefficients:')
for name, coef in zip(df_encoded.columns, lr.coef_):
    print(f' {name}: {coef:.2f}')

# **Step 7: PREDICTIONS AND RESIDUALS**
y_pred = lr.predict(df_encoded)
residuals = df[outcome] - y_pred

# **Step 8: RESIDUALS VS PREDICTED**
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Charges')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()

# **Step 9: Q-Q PLOT**
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

# **Step 10: HISTOGRAM OF RESIDUALS**
plt.hist(residuals, bins=30, edgecolor='k')
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# **Step 11: SHAPIRO TEST**
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f}")

# **Step 12: REFIT MODEL USING STATSMODELS**
X_sm = sm.add_constant(df_encoded).astype(float)
y_sm = df[outcome].astype(float)
model = sm.OLS(y_sm, X_sm).fit()

# **Step 13: DURBIN-WATSON TEST**
dw = durbin_watson(model.resid)
print(f"Durbin-Watson statistic: {dw:.2f}")

# **Step 14: BREUSCH-PAGAN TEST**
bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_labels = ['Lagrange multiplier', 'p-value', 'f-value', 'f p-value']
bp_results = dict(zip(bp_labels, bp_test))
print("\nBreusch–Pagan Test Results:")
for key, val in bp_results.items():
    print(f" {key}: {val:.4f}")

# **Step 15: MULTICOLLINEARITY (VIF)**
vif_data = pd.DataFrame()
vif_data['feature'] = df_encoded.columns
vif_data['VIF'] = [variance_inflation_factor(df_encoded.values, i)
                   for i in range(df_encoded.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# **Step 16: OLS MODEL USING TRAIN/TEST SPLIT**
X = pd.get_dummies(df[predictors], drop_first=True).astype(float)
y = df[outcome].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
model_ols = sm.OLS(y_train, X_train_const).fit()
print(model_ols.summary())

# **Step 17: PREDICTIONS**
y_pred_train = model_ols.predict(X_train_const)
y_pred_test = model_ols.predict(X_test_const)

# **Step 18: RESIDUALS**
train_residuals = y_train - y_pred_train
test_residuals = y_test - y_pred_test

# **Step 19: MODEL EVALUATION**
from sklearn.metrics import mean_absolute_error
print("\nTraining Performance:")
print("MAE:", mean_absolute_error(y_train, y_pred_train))
print("RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("R²:", r2_score(y_train, y_pred_train))

print("\nTesting Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("R²:", r2_score(y_test, y_pred_test))

# **Step 20: RESIDUALS VS PREDICTED (TEST SET)**
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred_test, y=test_residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Charges (Test)')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (Test Set)')
plt.show()

# **Step 21: ASSUMPTION TESTS**

plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values (Predicted Charges)')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

plt.hist(residuals, bins=30, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Shapiro Test
shapiro_test = shapiro(residuals)
print(f"\nShapiro-Wilk p-value: {shapiro_test.pvalue:.4f}")

# Durbin-Watson
dw_stat = durbin_watson(model_ols.resid)
print(f"\nDurbin-Watson statistic: {dw_stat:.2f}")

# Breusch–Pagan
bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)
bp_results = dict(zip(bp_labels, bp_test))
print("\nBreusch–Pagan Test Results:")
for key, val in bp_results.items():
    print(f" {key}: {val:.4f}")

# VIF
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

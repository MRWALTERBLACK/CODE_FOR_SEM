# **Step 1: IMPORT PACKAGES**
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# **Step 2: LOAD DATASET**
df = pd.read_csv(r"C:\Users\MAYUR\Downloads\Salary_Data.csv")
df.head()

# **Step 3: DATA ANALYSIS**
df.describe()

plt.title('Salary Distribution Plot')
sns.histplot(df['Salary'], kde=True, color='salmon')
plt.show()

plt.scatter(df['YearsExperience'], df['Salary'], color='lightcoral')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

# **Step 4: SPLIT DATASET**
X = df.iloc[:, :1]
y = df.iloc[:, 1:]

# **Step 5: TRAIN-TEST SPLIT**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# **Step 6: TRAIN THE MODEL**
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# **Step 7: PREDICT RESULTS**
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# **Step 8: PLOT TRAINING RESULTS**
plt.scatter(X_train, y_train, color='lightcoral')
plt.plot(X_train, y_pred_train, color='firebrick')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['Predicted', 'Actual'], loc='best', facecolor='white')
plt.box(False)
plt.show()

# **Step 9: PLOT TEST RESULTS**
plt.scatter(X_test, y_test, color='lightcoral')
plt.plot(X_train, y_pred_train, color='firebrick')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['Predicted', 'Actual'], loc='best', facecolor='white')
plt.box(False)
plt.show()

# **Step 10: MODEL PARAMETERS**
print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
print(f'R² Score (Training): {regressor.score(X_train, y_train):.4f}')
print(f'R² Score (Testing): {regressor.score(X_test, y_test):.4f}')

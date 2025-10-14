# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# Step 2: Load Data
data = pd.read_csv(r"C:\Users\MAYUR\Downloads\diabetes.csv")
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Plain Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print("Accuracy (Plain Decision Tree):", accuracy_score(y_test, clf.predict(X_test)))

# Step 5: AdaBoost with Stumps
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_stump = AdaBoostClassifier(estimator=stump, n_estimators=100, learning_rate=0.5, random_state=42)
ada_stump.fit(X_train, y_train)
print("Accuracy (AdaBoost + Stumps):", accuracy_score(y_test, ada_stump.predict(X_test)))

# Step 6: AdaBoost with Deeper Trees
tree3 = DecisionTreeClassifier(max_depth=3, random_state=42)
ada_tree3 = AdaBoostClassifier(estimator=tree3, n_estimators=50, learning_rate=0.5, random_state=42)
ada_tree3.fit(X_train, y_train)
print("Accuracy (AdaBoost + depth=3 trees):", accuracy_score(y_test, ada_tree3.predict(X_test)))

# Step 7: Plot First Stump
first_stump = ada_stump.estimators_[0]
plt.figure(figsize=(10,6))
tree.plot_tree(first_stump,
               feature_names=X.columns,
               class_names=["Non-Diabetic", "Diabetic"],
               filled=True, rounded=True)
plt.show()

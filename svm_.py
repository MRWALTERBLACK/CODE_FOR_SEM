#1 import libraries
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics


# 2 Load dataset
df = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\iris.csv")
df = df.drop('Id',axis=1)
print(df.head())

# 
df['y'] = df['Species'].apply(lambda x:0 if 'setosa'in x.lower() else 1)

X = df[['PetalLengthCm','PetalWidthCm']].values
y = df[['y']].values

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# linear SVM
clf_lin = svm.SVC(kernel='linear')
clf_lin.fit(X_train,Y_train)
y_pred_lin = clf_lin.predict(X_test)

#NON linear SVM
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,Y_train)
y_pred_rbf = clf_rbf.predict(X_test)

print("Accuracy",metrics.accuracy_score(Y_test,y_pred_lin))
print("Precision",metrics.precision_score(Y_test,y_pred_lin))
print("recall",metrics.recall_score(Y_test,y_pred_lin))

print("Accuracy",metrics.accuracy_score(Y_test,y_pred_rbf))
print("Precision",metrics.precision_score(Y_test,y_pred_rbf))
print("recall",metrics.recall_score(Y_test,y_pred_rbf))

# visualisation
y_all_pred = clf_lin.predict(X)
plt.scatter(X[:,0],X[:,1], c= y_all_pred,cmap='coolwarm',edgecolors='k')
plt.xlabel("Petal lenght")
plt.ylabel("petal width")
plt.show()
#Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#load the dataset
df  = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\housing_cal.csv",usecols=['longitude','latitude','median_house_value'])
print(df.head())

#Raw data plot
sns.scatterplot(data=df,x='longitude',y='latitude',hue='median_house_value')
plt.title("Scatter Plot for RAW data")
plt.show()

#train test split
X_train , X_test,y_train,y_test = train_test_split(
    df[['longitude','latitude']],
    df[['median_house_value']],
    test_size=0.33,random_state=0)

#plot
sns.scatterplot(x=X_train['longitude'],y=X_train['latitude'])
plt.xlabel("Longitude")
plt.ylabel("latitude")
plt.title("PLOT AFTER TRAIN TEST SPLIT")
plt.show()

# --- Standardize ---
scalar = preprocessing.StandardScaler()
X_train_norm = scalar.fit_transform(X_train)

plt.scatter(X_train_norm[:,1],X_train_norm[:,0])
plt.title("PLOT AFTER STANDARDIZE")
plt.show()

#kmeans
kmeans = KMeans(n_clusters=3,random_state=0,n_init='auto')
kmeans.fit(X_train_norm)
X_train['Cluster'] = kmeans.labels_

sns.scatterplot(data=X_train,x='longitude',y='latitude',hue=kmeans.labels_)
plt.title("k_means cluster plot")
plt.show()

# --- Silhouette for k=2..7 ---
K = range(2,8)
fits = []
scores = []

for k in K:
    model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X_train_norm)
    fits.append(model)
    scores.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

sns.lineplot(x=K, y=scores, marker='o')
plt.title("Silhouette score vs number of clusters")
plt.show()

# --- Compare different clusterings ---
sns.scatterplot(x=X_train['longitude'], y=X_train['latitude'], hue=fits[0].labels_, s=10)
plt.title("KMeans clustering (k=2)")
plt.show()

sns.scatterplot(x=X_train['longitude'], y=X_train['latitude'], hue=fits[1].labels_, s=10)
plt.title("KMeans clustering (k=3)")
plt.show()

sns.scatterplot(x=X_train['longitude'], y=X_train['latitude'], hue=fits[2].labels_, s=10)
plt.title("KMeans clustering (k=4)")
plt.show()

sns.scatterplot(x=X_train['longitude'], y=X_train['latitude'], hue=fits[3].labels_, s=10)
plt.title("KMeans clustering (k=5)")
plt.show()


# Step 2: Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 3: Load Data
data = pd.read_csv(r"C:\Users\MAYUR\Downloads\NetflixOriginals.csv", encoding='latin1')
data.sample(2)
print(data.head())

# Step 4: Plot Distribution
fig, ax = plt.subplots(1,1,figsize=(16, 8))
sns.histplot(data['Runtime'], kde=True)
plt.title("Distribution of Runtime")
plt.show()

# Step 5: One Sample T-Test (h_null = 91.3)
h_null = 91.3
test_result, pval = stats.ttest_1samp(data['Runtime'], h_null)
print(f"Mean of Runtime: {np.mean(data['Runtime'])}")
if pval < 0.05:
    print(f"Reject Null Hypothesis: Mean differs from {h_null}")
else:
    print(f"Fail to Reject Null: Mean equals {h_null}")
print(test_result, pval)

# Step 6: One Sample T-Test (h_null = 93)
h_null = 93
test_result, pval = stats.ttest_1samp(data['Runtime'], h_null)
if pval < 0.05:
    print(f"Reject Null Hypothesis: Mean differs from {h_null}")
else:
    print(f"Fail to Reject Null: Mean equals {h_null}")
print(test_result, pval)

# Step 7: Two Sample T-Test (dummy = [10, 60])
dummy = [10, 60]
test_result, pval = stats.ttest_ind(data['Runtime'], dummy)
if pval < 0.05:
    print("Reject Null Hypothesis: Means are different")
else:
    print("Fail to Reject Null: Means are identical")
print(test_result, pval)

# Step 8: Plot Comparison
fig, ax = plt.subplots(1,1,figsize=(16, 8))
sns.histplot(data['Runtime'], kde=True)
sns.histplot(dummy, kde=True, color='orange')
ax.legend(['Runtime', 'Dummy'])
plt.show()

# Step 9: Two Sample T-Test (dummy = [10, 60, 99, 110, 160])
dummy = [10, 60, 99, 110, 160]
test_result, pval = stats.ttest_ind(data['Runtime'], dummy)
if pval < 0.05:
    print("Reject Null Hypothesis: Means are different")
else:
    print("Fail to Reject Null: Means are identical")
print(test_result, pval)

# Step 10: Plot Final Comparison
fig, ax = plt.subplots(1,1,figsize=(16, 8))
sns.histplot(data['Runtime'], kde=True)
sns.histplot(dummy, kde=True, color='orange')
ax.legend(['Runtime', 'Dummy'])
plt.title("T-Test: Mean Comparison", size=22)
plt.show()

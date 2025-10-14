# Chi-Square Test Script (Clean Version)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
dataset = pd.read_csv(r"C:\Users\MAYUR\Downloads\children anemia.csv")
dataset.rename(columns={"Anemia level...8": "Anemia level"}, inplace=True)

# Step 2: Display columns
for col in dataset.columns:
    print(col)

# Step 3: Create contingency table
selected_data = dataset[["Highest educational level", "Anemia level"]]
contingency_table = pd.crosstab(selected_data["Highest educational level"], selected_data["Anemia level"])
print(contingency_table)

# Step 4: Perform Chi-Square Test
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2_stat)
print("Degrees of Freedom:", dof)
print("P-Value:", p_val)
print("Expected Frequencies:\n", expected)

# Step 5: Observed and Expected Counts
observed_counts = contingency_table.values
print("Observed Counts:\n", observed_counts)
expected_counts_rounded = np.round(expected, 2)
print("Expected Counts (Rounded):\n", expected_counts_rounded)

# Step 6: Pearson Residuals
pearson_residuals = (contingency_table.values - expected) / np.sqrt(expected)
pearson_residuals_rounded = np.round(pearson_residuals, 2)
pearson_residuals_df = pd.DataFrame(pearson_residuals_rounded,
                                    index=contingency_table.index,
                                    columns=contingency_table.columns)
print("Pearson Residuals:\n", pearson_residuals_df)

# Step 7: Percentage Contribution to Chi-Square
contributions = ((contingency_table.values - expected) ** 2) / expected
percentage_contributions = 100 * contributions / chi2_stat
percentage_contributions_df = pd.DataFrame(np.round(percentage_contributions, 2),
                                           index=contingency_table.index,
                                           columns=contingency_table.columns)
print("Percentage Contributions to Chi-Square Statistic:")
print(percentage_contributions_df)

# Step 8: Heatmap Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(percentage_contributions_df,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Percentage Contribution'},
            linewidths=0.5,
            linecolor='gray')

plt.title("Percentage Contribution to Chi-Square Statistic")
plt.xlabel("Anemia Level")
plt.ylabel("Highest Educational Level")
plt.tight_layout()
plt.show()

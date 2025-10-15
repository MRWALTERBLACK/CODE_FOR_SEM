# -*- coding: utf-8 -*-
"""
One-Way ANOVA Example (Improved)
---------------------------------
This script simulates voter ages across racial groups
and tests whether their mean ages differ significantly.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# -----------------------------
# Step 1: Generate Random Data
# -----------------------------
np.random.seed(12)

races = ["Asian", "Black", "Hispanic", "Other", "White"]

# Generate random race data with probabilities
voter_race = np.random.choice(
    a=races,
    p=[0.05, 0.15, 0.25, 0.05, 0.50],
    size=1000
)

# Generate age data using Poisson distribution
voter_age = stats.poisson.rvs(loc=18, mu=30, size=1000)

# Create DataFrame
voter_frame = pd.DataFrame({
    "Race": voter_race,
    "Age": voter_age
})

# -----------------------------
# Step 2: Grouping by Race
# -----------------------------
groups = voter_frame.groupby("Race").groups

# Extract groups
asian = voter_age[groups["Asian"]]
black = voter_age[groups["Black"]]
hispanic = voter_age[groups["Hispanic"]]
other = voter_age[groups["Other"]]
white = voter_age[groups["White"]]

# -----------------------------
# Step 3: Perform One-Way ANOVA (SciPy)
# -----------------------------
f_stat, p_value = stats.f_oneway(asian, black, hispanic, other, white)
print("=== One-Way ANOVA using SciPy ===")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis: Significant difference found among groups.\n")
else:
    print("Fail to reject the null hypothesis: No significant difference found.\n")

# -----------------------------
# Step 4: Detailed ANOVA (StatsModels)
# -----------------------------
model = ols('Age ~ Race', data=voter_frame).fit()
anova_result = sm.stats.anova_lm(model, typ=2)

print("=== ANOVA Table (StatsModels) ===")
print(anova_result)

# Optional: print mean ages
print("\n=== Mean Age by Race ===")
print(voter_frame.groupby("Race")["Age"].mean())

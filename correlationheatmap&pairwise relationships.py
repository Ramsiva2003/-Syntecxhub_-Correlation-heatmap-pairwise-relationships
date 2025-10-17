# ----------------------------------------------
# 📦 Import Required Libraries
# ----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------
# 🧩 Step 1: Create a Sample Dataset
# ----------------------------------------------
np.random.seed(42)

# Generate 200 samples with correlated numerical data
data = {
    'Advertising_Spend': np.random.randint(1000, 10000, 200),
    'Product_Price': np.random.uniform(10, 100, 200),
    'Customer_Rating': np.random.uniform(1, 5, 200),
    'Website_Visits': np.random.randint(100, 10000, 200),
    'Units_Sold': np.random.randint(50, 1000, 200)
}

df = pd.DataFrame(data)

# Add a feature that’s somewhat correlated to others
df['Revenue'] = (df['Units_Sold'] * df['Product_Price']) + np.random.normal(0, 2000, 200)

print("Sample Data:")
print(df.head())

# ----------------------------------------------
# 🔢 Step 2: Compute Correlation Matrix
# ----------------------------------------------
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix)

# ----------------------------------------------
# 🔥 Step 3: Correlation Heatmap
# ----------------------------------------------
plt.figure(figsize=(8,6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# ----------------------------------------------
# 🔍 Step 4: Pairwise Relationships (Pairplot)
# ----------------------------------------------
sns.pairplot(df[['Advertising_Spend', 'Product_Price', 'Customer_Rating', 'Units_Sold', 'Revenue']],
             diag_kind='kde', corner=True)
plt.suptitle('Pairwise Relationships Between Key Variables', y=1.02)
plt.savefig('pairwise_relationships.png')
plt.show()

# ----------------------------------------------
# 🧾 Step 5: Summary of Findings
# ----------------------------------------------
# Identify strongest correlations
corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
corr_unstacked = corr_unstacked[corr_unstacked < 1.0]  # remove self correlations

strongest_positive = corr_unstacked.head(1)
strongest_negative = corr_unstacked.tail(1)

summary = f"""
Correlation Analysis Summary:
-----------------------------
The heatmap visualizes pairwise Pearson correlations between numerical variables.

Strongest Positive Correlation:
{strongest_positive.index[0][0]} and {strongest_positive.index[0][1]} → Correlation = {strongest_positive.values[0]:.2f}

Strongest Negative Correlation:
{strongest_negative.index[0][0]} and {strongest_negative.index[0][1]} → Correlation = {strongest_negative.values[0]:.2f}

Observations:
- Revenue is strongly positively correlated with Units_Sold and Product_Price, as expected.
- Advertising_Spend may show weak to moderate correlation with Units_Sold and Revenue.
- Customer_Rating tends to be independent from numerical marketing metrics.

Charts saved:
- correlation_heatmap.png
- pairwise_relationships.png
"""

print(summary)

# Save summary as text file (with UTF-8 encoding to avoid Unicode errors)
with open("correlation_analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\n✅ Task 3 Completed Successfully! Charts and summary saved in your project folder.")


"""
eda.py
------
Exploratory Data Analysis for the used-car dataset.
Produces summary stats + saves charts to outputs/.

Run:
  python3 eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/used_cars.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print(df.info())
print("=" * 60)
print("MISSING VALUES:\n", df.isnull().sum())
print("=" * 60)
print("PRICE STATS:\n", df["selling_price"].describe())

# 1. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], bins=40, kde=True, color="steelblue")
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.tight_layout()
plt.savefig("outputs/price_distribution.png", dpi=120)
plt.close()

# 2. Price vs Age
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="selling_price", alpha=0.4, hue="fuel_type")
plt.title("Selling Price vs Car Age")
plt.tight_layout()
plt.savefig("outputs/price_vs_age.png", dpi=120)
plt.close()

# 3. Price vs KM driven
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="km_driven", y="selling_price", alpha=0.4, color="darkorange")
plt.title("Selling Price vs KM Driven")
plt.tight_layout()
plt.savefig("outputs/price_vs_km.png", dpi=120)
plt.close()

# 4. Avg price by brand
plt.figure(figsize=(9, 5))
order = df.groupby("brand")["selling_price"].mean().sort_values(ascending=False).index
sns.barplot(data=df, x="brand", y="selling_price", order=order, estimator="mean", errorbar=None)
plt.title("Average Selling Price by Brand")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/price_by_brand.png", dpi=120)
plt.close()

# 5. Correlation heatmap (numeric columns)
plt.figure(figsize=(7, 6))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=120)
plt.close()

print("\nSaved 5 charts to outputs/")

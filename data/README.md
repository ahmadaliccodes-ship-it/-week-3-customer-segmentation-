# Data

## Source

This project uses the **Wholesale Customers** dataset from the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/292/wholesale+customers

## Files

- `raw/wholesale_customers.csv` — original dataset used by the project.
- `processed/scaled_features.csv` — log-transformed and standardized clustering features.
- `processed/labeled_customers.csv` — original customer records plus K-Means cluster labels.
- `processed/stability_summary.csv` — seed-variation and bootstrap stability results.

## Modeling fields

Clustering uses only these six annual-spend variables:

- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

`Channel` and `Region` are excluded from clustering and used only for post-hoc profiling.

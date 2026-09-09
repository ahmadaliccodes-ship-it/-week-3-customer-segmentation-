# Week 3 — Customer Segmentation

Unsupervised customer segmentation of wholesale customers using K-Means clustering, with hierarchical clustering as a cross-check and PCA for visualization.

## Objective

Segment 440 wholesale customers using annual spending across six product categories and identify interpretable customer groups that can support targeted business decisions.

## Workflow

1. Dataset setup and structure checks
2. Data-quality confirmation and exploratory analysis
3. Skewness, outlier, and multicollinearity diagnostics
4. Log transformation and standardization
5. K-Means / hierarchical clustering comparison and K selection
6. Final K-Means training and stability validation
7. Cluster profiling and business analysis
8. PCA visualization
9. Business interpretation and limitations
10. Final report assembly

## Final model

- Algorithm: K-Means
- Selected K: 2
- Clustering features: six annual-spend variables
- Transformation: `log1p`
- Scaling: `StandardScaler`
- Random state: 42
- `n_init`: 20

Channel and Region are held out from clustering and used only as post-hoc context.

## Repository structure

```text
week-3-customer-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── notebooks/
├── src/
├── figures/
├── reports/
└── tests/
```

## Run

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Run notebooks in numerical order from `01_project_setup.ipynb` through `10_report_assembly.ipynb`.

## Key result

The K=2 solution had the strongest silhouette score among the tested K values and was more stable than the K=4 alternative under the project's seed-variation and bootstrap checks. The resulting segmentation is best treated as a broad purchasing-behavior split rather than a definitive ground truth.

## Report

The final report is available at:

`reports/Wholesale_Customer_Segmentation_Report.pdf`

## Important reproducibility note

The notebooks are the original phase-by-phase analysis record. Reusable core functions are also provided in `src/` so the final repository has a cleaner separation between analysis notebooks and reusable code.

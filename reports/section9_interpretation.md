# 9. Interpretation, Business Implications & Limitations

## 9.1 Cluster Interpretation

Cluster 0 — "Horeca (1) Fresh/Frozen Buyers"
  Size: 252 customers (57.3% of the dataset).
  Dominant spend categories: Fresh and Frozen (means: 13973 and 3706).
  Channel composition: 96.8% Horeca (1).
  Mean total annual spend: 24,529 (lower of the two clusters).


Cluster 1 — "Retail (2) Grocery/Milk Buyers"
  Size: 188 customers (42.7% of the dataset).
  Dominant spend categories: Grocery and Milk (means: 14697 and 10346).
  Channel composition: 71.3% Retail (2).
  Mean total annual spend: 44,884 (highest of the two clusters).


## 9.2 Business Implications

1. EVIDENCE-BASED COMMERCIAL TARGETING:
   - Marketing could prioritize Fresh-led cross-selling for Cluster 0 because Fresh has the highest mean annual spend in the segment (13,973), followed by Frozen (3,706). The segment is also 96.8% Horeca (1) based on Channel.
   - Marketing could prioritize Grocery-led cross-selling for Cluster 1 because Grocery has the highest mean annual spend in the segment (14,697), followed by Milk (10,346). The segment is also 71.3% Retail (2) based on Channel.

2. RELATIVE CUSTOMER VALUE:
   - Cluster 1 has mean total annual spend of 44,884 versus 24,529 for Cluster 0. This supports prioritizing value-based tests, not an assumption that higher spend will produce a stronger promotional response.

[CAVEAT] These are hypotheses based on observed spending and channel composition. They are not causal claims about promotion response, pricing sensitivity, delivery frequency, or future sales.

## 9.3 Decision Framework

Cluster 0 — Horeca (1) Fresh/Frozen Buyers
252 customers (57.3% of customers).
Decision hypothesis: Prioritize offers aligned with the segment's two highest mean-spend categories (Fresh and Frozen) and validate response using customer-level outcomes before operationalizing the strategy.

Cluster 1 — Retail (2) Grocery/Milk Buyers
188 customers (42.7% of customers).
Decision hypothesis: Prioritize offers aligned with the segment's two highest mean-spend categories (Grocery and Milk) and validate response using customer-level outcomes before operationalizing the strategy.

These recommendations are decision hypotheses, not causal claims. The dataset is observational and does not measure promotion response, margin, delivery frequency, or customer lifetime value.

## 9.4 Limitations

1. SMALL DATASET: only 440 customer records. Cluster boundaries and profile statistics (means, ANOVA F-values) are estimated from a relatively small sample and may not generalize to a larger customer base without re-validation.

2. SINGLE TIME-SNAPSHOT DATA: spend values are annual totals with no seasonality or time dimension. A customer's segment could shift across a year (e.g. a Horeca account's Fresh spend may spike seasonally) — this analysis cannot detect or account for that.

3. NO DEMOGRAPHIC/FIRMOGRAPHIC DATA: beyond Channel and Region, no other customer attributes (business size, years as customer, order frequency, etc.) are available, limiting how deeply the 'why' behind each cluster's spend pattern can be explained.

4. K-MEANS' SPHERICAL-CLUSTER ASSUMPTION: K-Means is most naturally suited to compact, roughly spherical clusters under Euclidean distance. The observed structure may not perfectly meet this assumption. The dendrogram (Phase 5) and residual skew after log-transform (Phase 4) suggest the true spend distribution may not perfectly fit this assumption, so some customers near the cluster boundary (visible in Phase 8's PCA scatter) may be ambiguously assigned.

5. LOG-TRANSFORM INTERPRETABILITY: the log1p transform (Phase 4) was necessary to stabilize variance for clustering, but it changes the geometry of distances — a given gap in log-space does not correspond to a constant dollar gap in raw spend. Business stakeholders reading 'distance between clusters' should refer to the raw mean/median tables (Phase 7), not transformed-space distances, for dollar-denominated conclusions.

6. SELECTED SOLUTION IS COARSE: the selected K is a low-dimensional segmentation, and its silhouette relative to an ideal separation and strong Channel alignment mean it may capture a broad business split rather than multiple fine-grained segments. This means the segmentation may be relatively coarse — it captures the dominant Horeca-vs-Retail-like spend split but may be masking finer sub-segments that this dataset's size/structure cannot reliably resolve with K-Means.

## 9.5 Conclusion & Recommended Next Steps

This analysis segmented 440 wholesale customers into 2 stable K-Means clusters based on annual spend across 6 product categories. The selected solution had the strongest K-Means silhouette among the stability-qualified candidates and passed the project's seed-variation and bootstrap stability checks.

  - Cluster 0: "Horeca (1) Fresh/Frozen Buyers"
  - Cluster 1: "Retail (2) Grocery/Milk Buyers"

The clusters align strongly with the pre-existing Channel variable even though Channel was excluded from clustering. This is best treated as post-hoc contextual evidence of business relevance, not external validation. The solution should therefore be interpreted as a broad spend segmentation that substantially separates Retail-dominant and Horeca-dominant purchasing patterns.

RECOMMENDED NEXT STEPS:
  1. Re-run the pipeline on more recent and larger customer data, ideally with monthly or transaction-level history, to test whether the segmentation persists over time.
  2. Validate the business hypotheses using order frequency, delivery, promotion-response, margin, and customer-tenure data.
  3. Investigate finer segmentation with additional algorithms only after validating whether K>2 solutions remain stable.
  4. Treat the current segments as decision-support hypotheses rather than fixed customer categories.

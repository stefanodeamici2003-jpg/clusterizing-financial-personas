# clusterizing-financial-personas — Unsupervised ML on Mixed-Type Data

> Segmenting 5,000 Italian bank clients into actionable personas using
> Gower Distance, K-Medoids, t-SNE, PCA, and LVQ1.

---

## Purpose

Retail banks manage heterogeneous client portfolios where customers differ across
financial capacity, investment behavior, demographics, and digital propensity.
Treating all clients uniformly leads to inefficient product allocation and missed
revenue opportunities.

This project applies unsupervised machine learning to discover natural groupings
within a portfolio of 5,000 Italian bank clients — with no predefined labels —
and translate those groupings into four interpretable business personas. Each persona
is characterized by a distinct financial profile, behavioral pattern, and demographic
composition, directly informing product strategy, marketing campaigns, and advisory
services.

---

## Methodology

The central challenge is that the dataset contains both **categorical** variables
(Job, Investments, Area, Gender, CitySize) and **numerical** variables (Income,
Wealth, Debt, FinEdu, Digital, ESG, and others). Standard algorithms assume a
single data type — this project addresses the mixed-type problem end to end.

### Gower Distance
Standard Euclidean distance is unsuitable for mixed data: encoding categorical
variables as integers implies ordinal relationships that do not exist.
Gower distance resolves this by computing a feature-specific sub-distance per
variable — normalized absolute difference for numericals, binary match/mismatch
for categoricals — and averaging into a single dissimilarity in [0, 1].
The result is a precomputed 5000×5000 matrix passed directly to all downstream
algorithms. Feature weights derived from Kruskal-Wallis H-statistics were explored
to amplify the most discriminative variables.

### K-Medoids (PAM)
K-Medoids was selected over K-Means for two reasons: it accepts a precomputed
distance matrix, making it natively compatible with Gower; and each cluster center
— the *medoid* — is an actual data point, guaranteeing full interpretability.
The PAM variant with greedy initialization (`init='build'`) was used to reduce
sensitivity to starting conditions. The optimal number of clusters was selected
by jointly evaluating the elbow method (inertia) and silhouette score across
k ∈ [2, 8], yielding k = 4.

### t-SNE and PCA
Two complementary dimensionality reduction techniques serve different purposes.
t-SNE operates on the precomputed Gower matrix to produce a 2D visualization
that preserves local neighborhood structure — used to visually confirm cluster
separation. PCA is applied on the encoded feature matrix to extract linear variance
structure and feature loadings, enabling interpretation of which variables drive
the principal components.

### Operational Validation: KNN and LVQ1
Once cluster labels were assigned, a supervised classifier was trained to validate
that the segments are learnable and to enable real-time classification of new
clients without recomputing the full Gower matrix. Both KNN (k=1) and a custom
LVQ1 implementation were evaluated. KNN achieved **88.2% accuracy**; LVQ1 reached
81.4%, with the gap quantifying the information cost of compressing each segment
into a single prototype vector.

---

## Key Insights

**Four personas were identified:**

| Cluster | Label | Key Signals |
|---------|-------|-------------|
| 0 | Mass Passive | Middle-income, passive or no investment behavior |
| 1 | Mass Active | Middle-income, active PAC investors, higher FinEdu & Digital |
| 2 | Elderly Conservatives | Oldest group (~79y), near-zero debt, low digital, high ESG |
| 3 | Young Affluent Investors | Youngest & wealthiest (~50y), high FinEdu, Digital, Luxury |

**C0 vs C1 — a behavioral gap, not a financial one.**
The two mass segments share nearly identical financial profiles (income ≈ 0.55,
wealth ≈ 0.56). The real discriminant is investment behavior: C0 clients do not
invest or invest passively; C1 clients invest actively through PAC — planned,
recurring contributions. A gender skew between the two clusters was investigated
and found to be spurious (Chi-square p = 0.46), confirming it is a geometric
artifact of the high-dimensional Gower space rather than a meaningful signal.

**Statistical robustness.**
Kruskal-Wallis tests confirm that all numerical variables significantly differentiate
the clusters (p < 0.001). The strongest discriminants are Luxury (H = 2128),
Digital (H = 2114), and FinEdu (H = 1821). Chi-square tests confirm the same for
all categorical variables. The 88.2% KNN accuracy further validates that cluster
boundaries are stable and learnable.

---

## How to Run

**Requirements**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn gower kmedoids
```

**Execution**
Open `BankClients_Segmentation.ipynb` in Jupyter and run all cells
sequentially. Note that the Gower matrix computation (~1–2 min) and t-SNE
(~2–3 min) are the most time-intensive steps. Cells containing these computations
can be skipped on subsequent runs if the results are already stored.

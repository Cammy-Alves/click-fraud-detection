# Ad Click Fraud Detection — A Machine Learning Approach

Detection of fraudulent clicks in Google Ads campaigns using five supervised machine learning algorithms on real data from a Brazilian marketing agency.

---

## Problem

A Brazilian digital marketing agency observed that many of the clicks it paid for on Google Ads never turned into an actual session on the client's website. In some accounts, up to 32% of the clicks were classified as invalid by third-party auditing tools, consistent with non-human or low-quality traffic (bots, click farms).

This project reproduces that classification with an open, auditable machine learning pipeline. Traffic Guardian labels are used as ground truth. The output is a ranked list of suspicious IPs that the agency can review and add to the Google Ads exclusion list.

## Result

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| SVM | 0.977 | 0.941 | **0.955** | 0.948 | 0.994 |
| Random Forest | 0.977 | 0.970 | 0.924 | 0.946 | 0.995 |
| KNN | 0.976 | 0.965 | 0.924 | 0.944 | 0.972 |
| Gradient Tree Boosting | **0.980** | **0.982** | 0.924 | **0.952** | 0.997 |
| XGBoost | 0.978 | 0.972 | 0.926 | 0.949 | **0.997** |

Gradient Tree Boosting was chosen for the best composite balance across accuracy, precision and F1. SVM edges out the others on recall alone, but GTB delivers the highest precision and F1 while keeping recall in the top tier.

The most predictive feature agreed upon by all five models is `Is_primary_click` — whether the click was the first action of the user's session. Non-primary clicks correlate almost perfectly with fraud (correlation with target = -0.95). Aggregate features (`Clicks_per_IP`, `Clicks_per_User_ID`) come second.

## Why these five algorithms

The mix was chosen to cover three families of decision boundaries in one comparison:

- Distance-based: KNN
- Margin-based: SVM
- Tree-based ensembles: Random Forest (bagging), Gradient Tree Boosting (boosting), XGBoost (regularised boosting)

Comparing across families shows whether the problem is intrinsically easy (all families win), whether it needs non-linearity (only trees win), or whether careful margin calibration matters (SVM leads). For this dataset, tree-based models take four of the top five metrics, which is what the literature on click fraud detection also reports.

## What surprised me during the analysis

`Is_primary_click` dominates so strongly that every other feature is essentially a rounding error in some models (importance ≈ 0.92 in XGBoost, 0.88 in GTB). This is a real signal, but it also means the model's performance is largely tied to this one column. If the upstream tracking that populates `Is_primary_click` ever changes, the model degrades sharply — a fact worth flagging to whoever operates the pipeline.

The rare-brand tail was full of near-perfect fraud rates (POCO 67%, Googlebot 100%, OkHttp 100%) but with very small counts. Grouping these into "Outros" was the right call — keeping them separate would have taught the model to memorise labels rather than to generalise the underlying pattern.

SMOTE was chosen over `class_weight` because it preserved the interpretability of the count-based features. Class weighting distorts the coefficients in ways that make the feature importance harder to defend to a non-technical stakeholder.

## Business impact

Google Ads already refunds a portion of the invalid clicks automatically, so this project does not promise direct R$ savings against those refunds. What it delivers is different:

- An independent auditing layer, so the agency and its clients do not rely solely on the platform that also sells the clicks.
- Visibility of fraud patterns per vertical (Client A has 2.3× the fraud rate of Client B), which becomes negotiation material when reviewing campaign mix and budget allocation.
- A reproducible pipeline that can be re-pointed, with re-labeling, to any similar detection problem: spam, trial abuse, bot mitigation, or fraud in other ad networks.

## In production

The current pipeline runs in batch mode over exported logs. A production-grade version would follow this pattern:

1. Ingestion. Nightly pull of clicks from the Google Ads API, stored as parquet in cloud storage.
2. Feature store. `Clicks_per_IP` and `Clicks_per_User_ID` computed over rolling 24h and 7d windows.
3. Scoring. Batch job scores yesterday's clicks with the persisted GTB model.
4. Action. IPs above a chosen probability threshold (initially 0.7, tuned quarterly with the agency) are exported to the Google Ads IP-exclusion list via API.
5. Monitoring. Track (a) the fraction of clicks with `Is_primary_click = 0` per week — a sudden drop signals upstream tracking drift; (b) the effective fraud rate per client per week — a sudden shift triggers investigation; (c) precision on manually reviewed IPs — the ground for retraining decisions.
6. Retraining. Full retrain quarterly, or ad-hoc if drift is detected.

## Scope

The ground truth is inherited from Traffic Guardian. The model reproduces that labeling. It does not replace or claim to outperform Google's own invalid-click detector.

The sample is restricted to two clients from a single agency across three months (January to March 2024). Generalisation to other verticals or time periods was not validated.

## Stack

Python 3.10+, pandas, numpy, scikit-learn (SVM, Random Forest, KNN, GradientBoosting, GridSearchCV, permutation importance), xgboost, imbalanced-learn (SMOTE), matplotlib, seaborn, joblib.

## Methodology

The work follows CRISP-DM:

1. Business Understanding. Agency context and problem definition.
2. Data Understanding. 13,098 clicks, 14 features, roughly 22% positive class (fraud).
3. Data Preparation. Cleaning of `(not set)` rows, grouping of rare categories, feature engineering (`Clicks_per_IP`, `Clicks_per_User_ID`), label encoding, MinMax scaling, SMOTE on the training set only.
4. Modeling. Five algorithms tuned with `GridSearchCV` (scoring = `f1`, cv = 5).
5. Evaluation. Comparison of all five models across train, validation and test, plus confusion matrices.
6. Deployment. Flagged IPs exported to Excel, model persisted with `joblib`.

---

Developed as part of a Master's thesis at NOVA IMS (Lisbon, 2024) and refactored for public portfolio sharing.
Author: Camilla Alves.

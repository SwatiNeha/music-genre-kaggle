# 🎵 Music Genre Classification — Kaggle Competition

This repository contains my end‑to‑end pipeline for a **music genre classification** competition on Kaggle. It covers
**EDA**, **feature engineering**, **modeling**, and **final submission** choices — all built to be reproducible. Images
are included where helpful (see `figures/`), and results are reported for both **validation** and **Kaggle leaderboard**.

---
## Real-World Applications

Music genre classification might begin as a Kaggle challenge, but it has clear real-world impact once taken outside the competition setting.

In the streaming industry, platforms like Spotify or Apple Music could use such models to enrich their recommendation engines. By automatically tagging songs with genres, the system can curate personalized playlists and help users discover music they’d never have found otherwise. Playlists can even adapt in real time - from mellow acoustic tracks in the evening to high-energy pop for a workout.

For the music industry and marketing, genre classifiers make it possible to automatically manage massive music catalogs without relying on inconsistent human metadata. Record labels and distributors can track trends in listener behavior across genres, informing marketing campaigns and release strategies.

On the consumer side, apps can let users upload or record snippets and instantly identify the genre, recommending similar artists and tracks. Fitness and wellness apps could tap into the same models to sync playlists with activities, boosting engagement by matching the right sound to the right moment.

And in research and education, genre classification offers a benchmark problem for teaching audio feature engineering and machine learning. More broadly, the same pipeline can be adapted to related domains such as speech emotion recognition, podcast tagging, or even anomaly detection in healthcare or industrial audio.

---

## Data Overview

- **Task**: Predict the **genre** (multi‑class) of a track from audio/meta features.
- **Train/Test shape**: Provided by competition (Kaggle). See EDA for quality notes.

### Missingness (from initial inspection)
**Train**: `tempo` (7,501 “?”), `duration_ms` (10,114 “-1”), `artist_name` (9,943 empty)  
**Test**: `tempo` (1,874 “?”), `duration_ms` (2,515 “-1”), `artist_name` (2,471 empty)

> These anomalies were normalized (coerced to numeric or set to NaN) and then imputed (see preprocessing).

---

## Exploratory Data Analysis (EDA)

**Key patterns:**
1) **Strong correlation:** `loudness` ↔ `energy` (≈ **0.81**) → redundant signal; remove one to reduce leakage/instability.  
2) **Trade‑off:** `acousticness` is **negatively** correlated with `energy` (~ −0.68) and `loudness` (~ −0.70).  
3) **Mutual information:** High for **popularity**, **acousticness**, **speechiness**; **low** for `tempo`, `time_signature`, `duration_ms`.  
4) **Skew & outliers:** `speechiness` (skew 2.30, kurtosis 3.97), `liveness` (skew 1.72), `loudness` (skew −1.34) — heavy tails/outliers → prefer
   log/quantile/Box‑Cox over standardization alone.  
5) **Bimodality:** `instrumentalness` has two modes → useful to **bin** / **cluster** (e.g., GMM) for better class separation.

**Illustrations (placeholders):**
- Correlation heatmap → `figures/corr_heatmap.png`  
- Pairwise / KDE grid → `figures/pairplot_kde.png`  
- Skew/outlier diagnostics → `figures/boxplots_skewness.png`  
- Instrumentalness bimodality → included in above figure
(These are generated in the EDA notebook.)

---

## Preprocessing & Feature Engineering

1) **Type & value fixes**
   - `time_signature`: parsed/encoded (e.g., “04‑Apr” → **4/4**, “03‑Apr” → **3/4**; unknown retained)
   - `tempo`: coercion “?” → NaN (numeric) 
   - `duration_ms`: “−1” → NaN as invalid duration

2) **Imputation**
   - **Iterative Imputer** for numeric columns (captures feature relations vs. single‑stat fill).

3) **Dimensionality / redundancy**
   - Drop **`energy`** (highly collinear with `loudness`).

4) **Distribution fixes**
   - **Log / PowerTransformer (Box‑Cox)** for skewed features: `speechiness`, `liveness`, `loudness`.

5) **Bimodal features**
   - **GMM features** for `instrumentalness` (binary/categorical modes).
   - **Acousticness** binned via **3‑component GMM** → low/med/high (drop raw to avoid redundancy).

6) **Filtering / selection**
   - **Mutual Information** ranking → keep high‑MI (e.g., `popularity`, `acousticness`, `speechiness`), drop low‑MI (`tempo`,
     `time_signature`, `duration_ms`).
   - **RFE** (with Random Forest) → retain ~**15** most relevant features.

7) **Final modeling frame**
   - Drop IDs / non‑predictive text: `instance_id`, `track_name`, `track_id`, **`artist_name`** (for final CatBoost run), `energy`.
   - **Categoricals**: `key`, `mode` (encoded when needed).  
   - **ColumnTransformer**: `StandardScaler` (numeric), `OneHotEncoder` (categorical).  
   - Target `genre` encoded via `LabelEncoder`.

---

## Models & Iterations

**Baselines / classical**
- **Logistic Regression** → ~**61.75%** validation.
- **KNN** → ~**60%** validation.
- **SVC (RBF)** → ~**65%** validation.
- **Random Forest** (tuned) → **69%** validation; **65%** Kaggle.

**Boosting family**
- **GradientBoostingClassifier** → **76%** validation; **65.3%** Kaggle.
- **HistGradientBoosting** → **75.74%** validation.
- **CatBoostClassifier** (early try, 1000 iters) → **76.89%** validation; **66.7%** Kaggle.

**Ensembles**
- **VotingClassifier** (CatBoost + GBC + HistGB, soft) → **76.91%** validation; **69%** Kaggle.

**Neural network**
- **Keras/TensorFlow** (Dense + BatchNorm + Dropout, EarlyStopping + ReduceLROnPlateau) → **79.4%** validation;
  **79.42%** Kaggle.

**Final submission (selected)**
- **CatBoostClassifier** (500 iters; *without* encoding `artist_name`) → **81%** validation; **81.6%** Kaggle;
  **ROC‑AUC 98.44%**.  Chosen for best generalization + native categorical handling + efficiency.

**Visuals (placeholders):**
- Feature importances (CatBoost) → `figures/feature_importances_catboost.png`  
---

## Results Summary

| Group                  | Model                        | Val Acc. | Kaggle | Notes |
|------------------------|------------------------------|:--------:|:------:|-------|
| Baseline               | Logistic Regression          | 61.75%   |   –    | Simple, interpretable【21†source】 |
| Baseline               | KNN                          | 60%      |   –    | Distance‑based【21†source】 |
| Baseline               | SVC (RBF)                    | 65%      |   –    | Non‑linear margin【21†source】 |
| Tree Ensemble          | Random Forest (tuned)        | 69%      | 65%    | Robust; FI available【21†source】 |
| Boosting               | GradientBoostingClassifier   | 76%      | 65.3%  | Strong baseline【21†source】 |
| Boosting               | HistGradientBoosting         | 75.74%   |   –    | Fast/Scalable【21†source】 |
| Boosting (CatBoost v1) | CatBoost (1000 iters)        | 76.89%   | 66.7%  | Initial config【21†source】 |
| Ensemble               | Voting (CBC+GBC+HistGB)      | 76.91%   | 69%    | Soft voting【21†source】 |
| Neural Net             | Keras Dense BN+Dropout       | 79.4%    | 79.42% | EarlyStopping, LR schedule【21†source】 |
| **Final**              | **CatBoost (500 iters)**     | **81%**  | **81.6%** | **ROC‑AUC 98.44%**; no `artist_name` encoding【21†source】 |

---

## Why CatBoost in the End?

- **Top accuracy** on both validation and leaderboard.  
- **Native categorical** support (no sparse blow‑ups from one‑hot).  
- **Interpretable enough** via **feature importance**; can add **SHAP** for local explanations if needed.  
- **Efficient** (500 iters) and easy to deploy.

> Notes on interpretability trade‑offs vs. KNN/Simple models are discussed in the report; SHAP/LIME can assist deployment
explanations if required.

---

## Reproducibility

- Set global **random seed** for NumPy / frameworks.  
- Notebooks/scripts included to generate **figures** and **submissions**.  
- Pipeline stages: **EDA → Preprocess/Impute → Feature‑eng → Select → Train → Validate → Submit**.

**Repo layout**
```
music-genre-kaggle/
├─ data/                  # train.csv / test.csv (Kaggle)
├─ files_created          # intermediate files created during training
├─ EDA.ipynb
├─ Preprocessing.ipynb
├─ Model.ipynb
├─ figures/               # plots referenced in README
└─ README.md              # this file
```


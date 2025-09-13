# Packet Anomaly Detection — Supervised Learning Flow

A complete, reproducible machine-learning workflow for **cybersecurity packet anomaly detection**.
This repository follows an academic assignment spec and demonstrates a clean ML flow from **data → EDA → experiments (5-fold CV + grid search) → final training → test evaluation**.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Environment & Setup](#environment--setup)
* [Repository Structure](#repository-structure)
* [Reproducibility](#reproducibility)
* [Methodology](#methodology)

  * [Phase 1 — Data & EDA](#phase-1--data--eda)
  * [Phase 2 — Experiments (Grid Search + 5-Fold CV)](#phase-2--experiments-grid-search--5-fold-cv)
  * [Phase 3 — Best Configuration Freeze](#phase-3--best-configuration-freeze)
  * [Phase 4 — Final Training](#phase-4--final-training)
  * [Phase 5 — Test Evaluation](#phase-5--test-evaluation)
* [Results](#results)
* [Known Issues & Workarounds (Windows/Python 3.13)](#known-issues--workarounds-windowspython-313)
* [Next Steps](#next-steps)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Project Overview

**Goal:** build a robust **multi-class classifier** that distinguishes between:

* `no_anomaly`
* `suspicious`
* `anomaly`

with a heavy class imbalance toward `no_anomaly`. The project emphasizes **experiment quality** (proper cross-validation, unbiased metrics, systematic hyperparameter search) and **clear, reproducible reporting** suitable for both coursework and real-world IDS (Intrusion Detection System) baselines.

**Primary metric:** **Macro-F1** (averages F1 across classes and is appropriate for imbalance).

---

## Dataset

Two CSV files are provided:

* `packets_train.csv`
* `packets_test.csv`

The label column is **`label`** with the classes listed above. In our EDA, the training split showed a **very strong imbalance** (\~99% `no_anomaly`), which drives several design decisions (e.g., Macro-F1 scoring, class-weighted models).

> **Note:** We use the provided train/test **as-is** (no re-splitting).

---

## Environment & Setup

We recommend Python **3.11 or 3.12** (Windows users on Python 3.13 may need workarounds; see below).

```bash
# 1) Create a virtual environment
python -m venv .venv
#    Activate it
#    - Windows (PowerShell): .venv\Scripts\Activate.ps1
#    - macOS/Linux:          source .venv/bin/activate

# 2) Install core dependencies
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib

# Optional (only if permitted by your course rules)
pip install imbalanced-learn
```

**Jupyter** (if needed):

```bash
pip install notebook
jupyter notebook
```

Open the notebook: **`Assignment_supervised_learning_flow.ipynb`** and run cells top-to-bottom.

---

## Repository Structure

```
.
├── Assignment_supervised_learning_flow.ipynb   # Main end-to-end notebook
├── packets_train.csv                           # Provided training split
├── packets_test.csv                            # Provided test split
└── README.md                                   # You are here
```

---

## Reproducibility

* We use `random_state = 42` where applicable.
* Experiments are run with **5-fold Stratified CV** and **GridSearchCV**.
* The final pipeline is **frozen** (best params) and retrained on **all training data** before **single** test evaluation.
* All plots use **matplotlib** only (no seaborn/styles).

> On **Windows + Python 3.13**, default process-based parallelism may fail with `_posixsubprocess` errors. Use the **threading** backend or set `n_jobs=1`. Details below.

---

## Methodology

### Phase 1 — Data & EDA

1. **Load** `packets_train.csv` and `packets_test.csv` and **preview first 5 rows** of each.
2. **Identify label** (`label`) and **summarize class balance** (strongly imbalanced).
3. **Schema & missingness:** count numeric vs categorical, per-column missing%.
4. **Visualizations** (matplotlib only):

   * Class balance bar chart
   * Histograms for top-variance numeric features (e.g., `packet_length`, `ttl`, `timestamp`)
   * Correlation heatmap (subset of numeric features for readability)
   * Boxplot of a high-variance feature grouped by `label` (checks separability)

**Why:** clarifies data quality, confirms supervised setup, and informs preprocessing choices (scaling, selection, encoding).

---

### Phase 2 — Experiments (Grid Search + 5-Fold CV)

**Design principles**

* Wrap the full preprocessing in a **scikit-learn `Pipeline`** so each split is transformed independently (prevents leakage).
* Search across **feature-engineering knobs** × **models** × **hyperparameters** with **Macro-F1** scoring.

**Preprocessing & feature engineering**

* Optional **column drop** (e.g., `timestamp`)
* **Scaling**: `StandardScaler`, `MinMaxScaler`, or passthrough
* **OneHotEncoder** for categoricals
* **SelectPercentile(f\_classif)**: `{100, 50, 25}`

**Models**

* `LogisticRegression(class_weight='balanced')` with `C ∈ {0.1, 1, 10}`
* `LinearSVC(class_weight='balanced')` with `C ∈ {0.1, 1, 10}`
* `RandomForestClassifier(class_weight='balanced')` with grid over `n_estimators`, `max_depth`, `min_samples_split`

**Validation**

* **StratifiedKFold(n\_splits=5, shuffle=True, random\_state=42)**
* **GridSearchCV** with `scoring='f1_macro'`, `refit=True`

**Outputs**

* A tidy **CV results table** sorted by mean Macro-F1
* The **best params** and the refitted `best_pipe`

---

### Phase 3 — Best Configuration Freeze

* Record the **exact best parameters** from Phase 2 (pipeline + hyperparameters + feature selection and scaling choices).
* This “frozen” configuration becomes the **final pipeline** template for retraining.

---

### Phase 4 — Final Training

* Rebuild a **fresh clone** of the base pipeline using the **frozen best parameters**.
* **Retrain on all training data**.
* Keep this fitted estimator as `final_pipe`.

---

### Phase 5 — Test Evaluation

* Evaluate `final_pipe` **once** on the test split.
* Report:

  * **Macro-F1 (test)**
  * Full **classification report**
  * **Confusion matrix** (+ heatmap)
  * **First 5 predictions** (as required)

> We do **not** re-tune on the test set. The test results are for **unbiased, final reporting** only.

---

## Results

> Fill in these placeholders after running the notebook on your machine.

* **Best CV Macro-F1:** `X.XXXX`
  *(from Phase 2 GridSearchCV)*

* **Test Macro-F1:** `X.XXXX`
  *(from Phase 5 evaluation)*

**Best configuration (example template):**

```
dropcols__kw_args: {'columns_to_drop': ['timestamp']}
preprocess__num__scaler: StandardScaler()
select__percentile: 50
clf: RandomForestClassifier(...)
clf__n_estimators: 400
clf__max_depth: 20
clf__min_samples_split: 10
```

*(Replace with your actual `grid.best_params_` printout.)*

---

## Known Issues & Workarounds (Windows/Python 3.13)

If you see errors like `ModuleNotFoundError: No module named '_posixsubprocess'` triggered by joblib/loky:

* **Use threads instead of processes** for parallelism:

  ```python
  from joblib import parallel_backend
  with parallel_backend('threading', n_jobs=-1):
      grid.fit(X_train, y_train)
  ```
* Or set **serial** execution:

  ```python
  grid.set_params(n_jobs=1)
  grid.fit(X_train, y_train)
  ```
* Avoid nested parallelism in tree models (e.g., `clf__n_jobs=1`).
* Consider Python **3.11/3.12** and update packages:

  ```bash
  pip install -U scikit-learn joblib
  ```

---

## Next Steps

* **Imbalance handling:** experiment with class weights vs. resampling (e.g., `SMOTE`) **inside** CV only (if allowed by course rules).
* **Explainability:** add permutation importance or SHAP for feature attributions.
* **Temporal robustness:** time-based splits by day/pcap to check distribution shift.
* **Model variants:** try Gradient Boosting / XGBoost / LightGBM (only if permitted by the assignment).

---

## License

This project is for educational purposes. Add a license of your choice if you plan to publish the full repo.

---

## Acknowledgments

* Scikit-learn for pipelines, grid search, and metrics.
* Matplotlib for visualizations.
* Course staff for the assignment structure.

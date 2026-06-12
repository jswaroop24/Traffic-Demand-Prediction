# 🚦 Flipkart Gridlock Hackathon: Traffic Demand Prediction

> **Architecture:** Tri-Model Spatio-Temporal Ensemble

---

## 🧠 1. Explanation of the Approach

Our core strategy was to build a highly defensive, spatio-temporal machine learning pipeline capable of modelling complex traffic micro-patterns without overfitting the sparse data. Because traffic demand is highly skewed (frequent zeros mixed with extreme traffic spikes), predicting raw demand directly led to high variance. To solve this, we applied a **Logarithmic Target Transformation** (`np.log1p`), which mathematically penalized the model from overreacting to outliers, and then inverse-transformed the predictions (`np.expm1`).

To prevent "spatial leakage"—where a model memorizes the training data rather than learning underlying traffic patterns—we utilized a strict **5-Fold Cross-Validation Strategy**. All historical averages and target encodings were calculated strictly out-of-fold.

Finally, to maximize generalization on the hidden test set, we deployed a **Tri-Core Ensemble Architecture**. We blended the predictions of three distinct models:

* **LightGBM (Deep):** High complexity (`num_leaves=63`) to memorize hyper-specific street interactions.
* **LightGBM (Shallow):** Low complexity (`num_leaves=31`) to act as a regularizer learning broad city-wide trends.
* **CatBoost:** Utilizing its native Ordered Target Statistic algorithm to process spatial categories symmetrically, drawing fundamentally different mathematical boundaries than LightGBM.

---

## ⚙️ 2. Details of Feature Engineering

Feature engineering was the primary driver of our model's performance, focusing heavily on continuous mapping of time and space:

* **Spatio-Temporal Mapping:** The raw timestamp was converted into a continuous 1D integer index (`time_slot` from 0 to 95) representing 15-minute intervals. To allow the tree models to natively understand the transition from night to morning, we applied **Cyclical Sine and Cosine waves** to the time slots, mathematically connecting 11:45 PM to 12:00 AM.
* **Native Spatial Decoding:** We built a custom Base-32 binary decoder to translate the alphanumeric geohash strings directly into exact `latitude` and `longitude` coordinates, allowing the model to plot physical boundaries.
* **Geo-Statistical Baselines:** We extracted the historical mean and standard deviation (`std`) for every specific geohash out-of-fold. The standard deviation feature allowed the model to natively separate high-variance commercial zones from low-variance residential roads.
* **Smoothed Target Encoding (Regularization):** To map localized interactions, we created a combined `loc_slot` feature (Geohash + Time Slot). To prevent overfitting on sparse data (e.g., a street with only 1 or 2 historical records), we applied a smoothing formula (`alpha=20`) that mathematically pulled unreliable local averages toward the global city average.
* **Granular Imputation:** Missing environmental values (`Temperature` and `Weather`) were forward and backward-filled grouped by exact geohash locations to preserve the micro-climates of specific streets.

---

## 🛠️ 3. Tools Used

* **Programming Language:** Python 3.12
* **Data Manipulation & Processing:** `pandas`, `numpy`
* **Machine Learning Algorithms:** * `lightgbm` (LGBMRegressor for gradient boosting on decision trees)
    * `catboost` (CatBoostRegressor for symmetric tree building and native categorical handling)
* **Validation & Metrics:** `scikit-learn` (`KFold` for cross-validation, `r2_score` for local evaluation)
* **Development Environment:** Jupyter Notebook / Google Colab

---

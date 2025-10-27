# 🏠 Airbnb Price Optimization – Rio de Janeiro  

## 📘 Overview  
This project focuses on **predicting Airbnb listing prices in Rio de Janeiro** using advanced machine learning techniques.  
The goal is to help hosts **optimize pricing** based on listing features such as location, room type, number of reviews, and availability.

We used **log-transformed price prediction** and **XGBoost regression** with **hyperparameter tuning**, achieving a well-generalized and interpretable model.  

---

## 🎯 Objectives  
- Analyze Airbnb listings to uncover price patterns.  
- Build a predictive model for optimized pricing.  
- Evaluate the effect of location, reviews, and availability.  
- Use **XGBoost with log transformation** for better accuracy.  

---

## ⚙️ Tech Stack  
| Category | Tools & Libraries |
|-----------|------------------|
| Language | Python 🐍 |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit-learn, XGBoost |
| Feature Processing | ColumnTransformer, OneHotEncoder, StandardScaler |
| Evaluation | R², RMSE, MSE |

---

## 📁 Project Workflow  

### 1️⃣ Data Loading & Cleaning  
- Loaded `listings.csv` containing Airbnb data for Rio de Janeiro.  
- Checked for missing values and filled them appropriately:  
  - Numeric → median fill  
  - Categorical → `'Unknown'` fill  
- Combined rare categories in `neighbourhood` (kept top 20, rest grouped as `'Other'`).  

### 2️⃣ Feature Engineering  
- Removed redundant identifiers (`id`, `host_id`).  
- Derived new ratio-based features for availability and reviews.  
- Separated **categorical** and **numerical** features for encoding.

### 3️⃣ Encoding & Scaling  
- Used `ColumnTransformer`:
  - `OneHotEncoder` for categorical variables  
  - `StandardScaler` for numerical ones  
- Included both inside a **Pipeline** for reproducible transformation.  

### 4️⃣ Log Transformation  
- Applied `log1p()` on `price` to reduce skewness.  
- Inverse-transformed predictions later using `expm1()` for interpretability.

### 5️⃣ Model Building  
#### 🔹 Linear Regression (Baseline)
- Simple model for benchmark comparison.  
- Low R², high RMSE — indicating need for non-linear model.  

#### 🔹 XGBoost Regressor (Advanced)
- Captures non-linear relationships and feature interactions.  
- Performed **hyperparameter tuning** using GridSearchCV for:  
  - `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`

### 6️⃣ Evaluation  
| Metric | Value |
|--------|--------|
| **R² Score** | 0.63 |
| **MSE** | ~1.2 × 10⁶ |
| **RMSE** | ~1100 – 1300 |

---

## 📔 Notebook Structure & Explanation  

### 🧩 **1. Data Inspection**
Explored dataset using `.info()`, `.describe()`, and `.value_counts()`.  
Identified numeric vs categorical columns.

### 🔍 **2. EDA (Exploratory Data Analysis)**
- Price distributions (before and after log-transform)  
- Correlation heatmap between features  
- Average prices by room type and neighbourhood  

### 🧹 **3. Missing Value Handling**
- Median fill for numeric  
- Mode/“Unknown” fill for categorical  
- Combined rare categories for balanced encoding  

### 🧠 **4. Feature Engineering**
- Removed identifiers  
- Retained top 20 neighbourhoods  
- Created engineered ratios and availability features  

### 🔡 **5. Encoding & Preprocessing**
Pipeline used `OneHotEncoder` and `StandardScaler` to prepare model input safely.  

### ⚙️ **6. Model Training**
- Linear Regression: benchmark  
- XGBoost: tuned model using cross-validation  
- Chose best hyperparameters for final fit  

### 📈 **7. Model Evaluation**
- Compared predicted vs actual prices  
- Checked residual distributions  
- Feature importance visualized  

---

## 📊 Results & Visualizations  

| Visualization | Description |
|----------------|-------------|
| 📈 **Price Distribution (Before & After Log Transform)** | Shows how transformation normalizes the skewed target variable. |
| 🌍 **Average Price by Neighbourhood** | Reveals key geographic pricing trends. |
| 🏠 **Room Type vs Average Price** | Demonstrates that entire homes/apartments are priced significantly higher. |
| 🔥 **Feature Importance (XGBoost)** | Highlights which features drive pricing (neighbourhood, room type, availability). |
| 📉 **Residual Plot** | Displays how well predictions align with actual prices. |

*(You can insert your generated charts here as image files using `![](images/plot_name.png)`)*

---

## 💡 Key Insights  
✅ Neighborhood and room type are **top predictors** of price.  
✅ Listings with more reviews and availability show **stable pricing patterns**.  
✅ Log transformation drastically improved stability and accuracy.  
✅ XGBoost handled feature interactions better than linear models.  

---

## 🚀 Future Enhancements  
- Integrate **time-based data** (seasonality, festivals).  
- Deploy on **Streamlit** for real-time predictions.  
- Use **SHAP values** for deeper explainability.  
- Compare XGBoost with LightGBM or CatBoost.  

---

## 🧭 How to Run  

### ▶️ On Google Colab
1. Upload the notebook and `listings.csv`.  
2. Run all cells in sequence.  
3. The model will train and output metrics + plots.

### 🧮 Requirements
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## 🧑‍💻 Author

Vasu Agrawal
B.Tech Computer Engineering | NMIMS Shirpur
📍 National-Level Basketball Player | 💼 AI & ML Enthusiast
🔗 LinkedIn
 • GitHub

## 🏁 Final Summary

Using advanced regression techniques and feature engineering,
this project delivers an interpretable and data-driven model for Airbnb price optimization in Rio de Janeiro,
achieving R² ≈ 0.63 and RMSE ≈ 1200 after tuning and log transformation.

✨ This project demonstrates practical, real-world data science workflow — from raw data to business-ready insights.

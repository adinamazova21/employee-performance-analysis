# 🧠 Employee Performance Analysis and Prediction

This project predicts employee performance using regression and classification models based on HR and satisfaction data. It supports data-driven HR strategies for identifying high-performing employees and optimizing engagement.

---

## 🎯 Project Goals

- 🔢 **Regression Task**: Predict a continuous performance score using numerical and categorical features.
- 🟢 **Classification Task**: Categorize employees as high- or low-performers using a binary label based on median performance.
- 📊 **Insight Extraction**: Analyze satisfaction scores, work-life balance, salary, and job role features.

---

## 📂 Datasets

- `Employee.csv`: Demographics, work history, salary, department, etc.
- `PerformanceRating.csv`: Job satisfaction, work-life balance, self/manager ratings.

---

## 🔍 Features

- **Numerical**: Age, Salary, Years at company, Time since last promotion.
- **Categorical**: Gender, Department, Business Travel, Marital Status, etc.
- **Target Variables**:
  - `PerformanceRating`: Average of satisfaction and rating scores (regression).
  - `PerformanceClass`: 0 = Low performer, 1 = High performer (classification).

---

## 🧠 ML Models & Metrics

### 🔹 Regression
- **Linear Regression**
- **Random Forest Regressor**
- 📈 Metrics: MSE, RMSE, MAE, R²

| Model                  | R²    | RMSE   | MAE    |
|------------------------|-------|--------|--------|
| Linear Regression      | -0.0092 | 0.4990 | 0.3967 |
| Random Forest Regressor | -0.2332 | 0.5516 | 0.4401 |

🔍 *Both models had low R², indicating underfitting.*

---

### 🔸 Classification
- **Logistic Regression**
- **Random Forest Classifier**
- 📈 Metrics: Accuracy, Precision, Recall, F1

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 0.6036   | 0.2941    | 0.0095 | 0.0185   |
| Random Forest Classifier | 0.5261   | 0.3688    | 0.2971 | 0.3291   |

🔍 *Random Forest performed better across all metrics, especially in recall.*

---

## 📊 Visualizations

1. 📦 Boxplot: Distribution of satisfaction & rating metrics  
2. 🔥 Heatmap: Correlation between numeric features (e.g., tenure & promotions)  
3. 📊 Bar Charts: Regression and classification model comparison  
4. 🖼 Output saved as: `performance_visualization.png`

---

## 🛠 Technologies

- Python (pandas, scikit-learn, matplotlib)
- ColumnTransformer, OneHotEncoder, StandardScaler
- Pipelines for preprocessing + modeling

---

## 🚀 How to Run

1. Place `Employee.csv` and `PerformanceRating.csv` in the working directory.
2. Run `adina_mazova_iml_project.py`
3. Check outputs:
   - Printed metrics in console
   - Visualizations in PNG format

---

## 📌 Limitations

- Class imbalance (fewer high performers)
- Correlated features (e.g., tenure vs. promotion gap)
- No time series or sales productivity data

---

## 📎 Files Included

- `Employee_Rerformance_Adina_Mazova.py` — Python file
- `Dataset` — TFolder with datasets
- `performance_vizualization.png` — Performance vizualization image
- `README.md` — This file

---

## 📬 Contact

**Adina Mazova**  
📧 [adina.mazova@gmail.com](mailto:adina.mazova@gmail.com)  
🌐 [github.com/adinamazova21](https://github.com/adinamazova21)


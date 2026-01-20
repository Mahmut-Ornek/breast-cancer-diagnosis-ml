# 🧬 Breast Cancer Diagnosis Using Machine Learning

This project focuses on building an end-to-end supervised machine learning pipeline to classify breast tumors as **malignant** or **benign** using features extracted from fine needle aspirate (FNA) images.  
The goal is to develop accurate, reliable, and interpretable models to support early breast cancer diagnosis.
All experiments, evaluations, and results are conducted and documented in a single, fully reproducible Jupyter Notebook.

> 📌 **Course Project (COE305 – Machine Learning)**  
> **Role:** Primary contributor responsible for data preprocessing, exploratory data analysis, model development, evaluation, and hyperparameter tuning.

---

## 📊 Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset (Kaggle / UCI)
- **Samples:** 569
- **Features:** 30 numerical features
- **Target Variable:**  
  - Malignant → 1  
  - Benign → 0  

The dataset contains no missing values and represents real clinical measurements.  
Outliers were intentionally preserved to maintain medical validity.

---

## 🔍 Machine Learning Pipeline
1. Data cleaning and preprocessing  
2. Feature scaling using `StandardScaler`  
3. Exploratory Data Analysis (EDA)  
4. Baseline model training  
5. Ensemble learning and model comparison  
6. Hyperparameter tuning using cross-validation  
7. Model evaluation and interpretation  

---

## 🤖 Models Implemented

### Baseline Models
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  

### Ensemble Models
- Random Forest  
- Gradient Boosting  
- Stacking Classifier  

Model evaluation was performed using **Stratified 5-Fold Cross-Validation** to handle class imbalance.

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

For medical diagnosis tasks, particular attention was given to **recall and false negatives**, as misclassifying malignant cases has critical consequences.

---

## 🏆 Results Summary
- Best-performing models: **Gradient Boosting** and **Stacking**
- Achieved approximately **97.4% F1-score** after hyperparameter tuning
- Ensemble models consistently outperformed baseline classifiers
- Feature importance analysis highlighted tumor size, concavity, and texture as key predictors

---

## 📁 Project Structure

```text
.
├── data/
│   └── breast_cancer_data.csv
├── notebooks/
│   └── final_test.ipynb
└── README.md
```

---

## 🔧 Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab

---

## 🚀 Future Work
- SHAP analysis for improved model interpretability
- Testing on external clinical datasets
- Deployment as a simple prediction API
- User interface for real-time decision support

---

## 👤 Author
**Mahmut Örnek**

Computer Engineering Student
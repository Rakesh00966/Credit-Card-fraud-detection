# 💳 Credit Card Fraud Detection using Machine Learning [Link](https://github.com/Rakesh00966/Credit-Card-fraud-detection/blob/main/Untitled1.ipynb)

This project aims to detect fraudulent credit card transactions using various Machine Learning models.  
It involves **data preprocessing, feature analysis, model comparison, and performance evaluation** to identify the most accurate model for detecting fraud.

---

## 📊 Project Overview

Credit card fraud is a major issue in the financial sector.  
This project uses a dataset containing transactions made by European cardholders in September 2013.  
The dataset is **highly imbalanced**, with only 0.172% of transactions being fraudulent.

Our goal is to build a machine learning model that can **accurately detect fraudulent transactions** while minimizing false positives.

---

## 🧠 Key Features

- Cleaned and preprocessed the dataset (removed duplicates, handled imbalance)
- Performed **data visualization and correlation analysis**
- Implemented **feature scaling** using `StandardScaler`
- Compared multiple ML algorithms:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)  
  - XGBoost  
  - Naive Bayes
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to handle data imbalance
- Evaluated models using **Accuracy, Precision, Recall, and F1 Score**
- Visualized results with **Confusion Matrix and Performance Comparison**

---

## 🧩 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, imbalanced-learn |
| Model Saving | Joblib |

---

## 📁 Dataset Information

- **Dataset:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows:** 284,807 transactions  
- **Features:** 30 (including `Amount`, `Time`, and 28 PCA-transformed features)
- **Target Variable:** `Class` (0 = Non-fraud, 1 = Fraud)

---

## ⚙️ Project Workflow

1. **Importing Libraries**
2. **Loading and Exploring Dataset**
3. **Data Cleaning**
   - Handling duplicates  
   - Checking missing values  
   - Dropping irrelevant features (`Time`)
4. **Feature Scaling**
   - Standardized the `Amount` column
5. **Exploratory Data Analysis**
   - Distribution plots  
   - Correlation heatmaps  
   - Class imbalance visualization
6. **Data Splitting**
   - Train–Test Split with Stratification
7. **Handling Class Imbalance**
   - Applied **SMOTE** to balance classes
8. **Model Training**
   - Trained multiple ML algorithms
9. **Model Evaluation**
   - Compared models using Accuracy, Precision, Recall, F1-Score
10. **Visualization**
    - Confusion Matrix for the best model
11. **Model Saving**
    - Exported trained model using `joblib`

---

## 📈 Model Performance (Example Results)

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|-----------|---------|-----------|
| Random Forest | 0.999 | 0.961 | 0.925 | 0.942 |
| XGBoost | 0.998 | 0.957 | 0.919 | 0.938 |
| Logistic Regression | 0.985 | 0.791 | 0.712 | 0.749 |


---
## 🧾 Conclusion

- The dataset was **highly imbalanced**, making it challenging to detect fraud.
- After applying **SMOTE** and testing multiple models, **Random Forest and XGBoost** gave the best results.
- The project demonstrates how data preprocessing, feature scaling, and model selection can significantly improve fraud detection accuracy.

---

## 🚀 Future Improvements

- Implement deep learning models (e.g., LSTM, Autoencoders)
- Use real-time fraud detection with streaming data
- Optimize hyperparameters using GridSearchCV or Bayesian Optimization

---

## 🧑‍💻 Author

**Kuntigorla Rakesh**  
🎓 B.Tech in Computer Science Engineering  
📧 [rakeshkuntigorla2@example.com]  
🌐 [[LinkedIn URL](https://www.linkedin.com/in/rakesh572/)]  
🌐 [[Portfolio](https://rakeshportfolio.figma.site)]  
💡 *Aspiring Data Analyst with strong Python and ML skills*

---

---

⭐ **If you found this project helpful, please give it a star on GitHub!**


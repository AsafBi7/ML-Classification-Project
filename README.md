# Loan Approval Prediction - Machine Learning Project

This project aims to predict whether a loan application will be approved or rejected using various machine learning models. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and evaluation based on AUC and other performance metrics.

## 🔍 Project Objectives

- Predict loan approval using structured tabular data
- Compare model performance based on AUC (ROC curve)
- Recommend an optimal probability threshold based on business risk
- Demonstrate practical ML skills: data wrangling, feature engineering, and evaluation

## 📁 Files Included

- `Submission_group15.pdf` - Full project report (in Hebrew)
- `Submission_group15.ipynb` - Jupyter notebook with end-to-end ML pipeline
- `Submission_group15.csv` - Final submission file containing `customer_id` and predicted approval probabilities (`predict_prob`)
- `train.csv` - Dataset used for training 
- `test.csv` - Dataset used for evaluation

## 🧠 ML Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Artificial Neural Network (ANN)

## 📈 Key Techniques

- Missing value imputation and outlier treatment
- Feature transformation and encoding
- Model training with `GridSearchCV`
- AUC-based evaluation and threshold tuning
- `StratifiedKFold` cross-validation

## 🏆 Best Result

- **Model**: Random Forest  
- **AUC**: 0.9176  
- **Business-driven optimization**: False Positive Rate (FPR) minimized due to asymmetric risk of false approvals

## 📈 ROC Curve

This plot shows the ROC curve of the best-performing model (Random Forest) on the test set:
<img width="478" height="386" alt="image" src="https://github.com/user-attachments/assets/18857e29-f9e4-46ed-aea9-d9832e9ec82e" />


## 📊 Libraries & Tools

- Python (Pandas, NumPy, Scikit-learn)
- Matplotlib, Seaborn
- Jupyter Notebook

## 👤 Author

Asaf Biran  
B.Sc. Data Science Student, Tel Aviv University  
[LinkedIn Profile](https://www.linkedin.com/in/asaf-biran-97b92935b/)

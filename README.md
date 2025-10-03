# Customer Churn Prediction for Beta Bank

## 🎯 Project Objective

Beta Bank noticed a gradual but consistent churn of its customers. Recognizing that retaining existing customers is more cost-effective than acquiring new ones, the bank sought to build a model that could predict which customers are most likely to leave.

> The primary goal was to create a classification model with the highest possible **F1 score**, as this metric provides a balance between Precision and Recall, which is crucial for an imbalanced dataset like this one. The model needed to achieve a minimum **F1 score of 0.59** on the test set to be considered effective. The AUC-ROC metric was also measured for a comprehensive evaluation.

---

## 📊 Data Source

The dataset used for this project is a popular public dataset from Kaggle, which contains anonymized data on Beta Bank's customers.

- **Source:** [Bank Customer Churn Prediction - TripleTen](https://practicum-content.s3.us-west-1.amazonaws.com/datasets/Churn.csv)

--- 

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white)

---

## 🚀 Project Workflow

The project followed a systematic approach to address the classification problem with imbalanced data:

1.  **Data Preprocessing:** The data was prepared for training by handling missing values and converting categorical features (like `Geography` and `Gender`) into numerical format using **One-Hot Encoding**.
2.  **Class Imbalance Analysis:** An initial investigation revealed a significant class imbalance, with a much larger number of non-churning customers compared to churning ones.
3.  **Baseline Modeling:** Models (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`) were initially trained on the imbalanced data to establish a baseline performance.
4.  **Imbalance Handling:** Two techniques were applied to mitigate the class imbalance:
    -   **Class Weighting:** Adjusting the `class_weight` parameter in the models to penalize mistakes on the minority class more heavily.
    -   **Upsampling:** Increasing the number of instances in the minority class to balance the dataset.
5.  **Model Training & Hyperparameter Tuning:** The models were retrained on the balanced data. **GridSearchCV** was used to systematically find the optimal hyperparameters for the `RandomForestClassifier` to maximize the F1 score.
6.  **Final Evaluation:** The best-performing model was evaluated on the unseen test set to measure its F1 score and AUC-ROC.

---

## 📊 Results & Conclusion

> After comprehensive training and tuning, the **RandomForestClassifier** trained on **upsampled data** achieved the highest performance on the test set.
>
> -   **Final F1 Score:** **0.61**
> -   **Final AUC-ROC Score:** **0.85**

💡 The final model successfully surpassed the required F1 score threshold of 0.59. The comparison between the F1 and AUC-ROC scores confirmed the model's strong predictive power, demonstrating its ability to effectively identify customers at risk of churning while managing the challenge of a class imbalance.

---

## 📁 Project Context

This project was completed as part of the **TripleTen Data Science Bootcamp**. The accompanying Jupyter Notebook (`.ipynb`) includes the original feedback and comments from my project reviewer, showcasing my ability to apply advanced techniques like class imbalance handling and hyperparameter tuning to solve real-world business problems.

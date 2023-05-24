# Phase-3-Project

## Patient Treatment Classification Project

### Table of Contents
1. [Overview](#overview)
2. [Business Understanding](#business-understanding)
3. [Business Objectives](#business-objectives)
4. [Project Goal](#project-goal)
5. [Data Understanding](#data-understanding)
   - [Target and Features](#target-and-features)
6. [Data Processing](#data-processing)
7. [Explanatory Data Analysis](#explanatory-data-analysis)
8. [Modelling](#modelling)
9. [Evaluation](#evaluation)
10. [Conclusion](#conclusion)


### Overview<a id="overview"></a>

This project focuses on developing a machine learning model that classifies patient treatment as in-care or out-care based on their laboratory test results. The dataset comprises electronic health records collected from a private hospital in Indonesia.

The objective is to provide a predictive tool for healthcare professionals that automates the process of categorizing patients.By analyzing attributes such as haematocrit, haemoglobins, erythrocyte count, leucocyte count, thrombocyte count, MCH, MCHC, MCV, age, and gender, the model predicts whether a patient should be classified as in-care or out-care.

The model's predictions assist healthcare providers in making informed decisions regarding the appropriate level of care required for each patient. This optimization of resource allocation improves efficiency, reduces costs, and enhances the overall patient experience.


### Business Understanding<a id="business-understanding"></a>

The hospital aims to automate the process of categorizing patients as in-care or out-care using their laboratory test results. The objective is to provide healthcare professionals with a predictive tool that assists in making informed decisions regarding patient treatment and care plans. By accurately classifying patients, the hospital can optimize resource allocation, reduce costs, and enhance the overall patient experience.

### Business Objectives<a id="business-objectives"></a>

- Improve Patient Care Classification: Automate the process of categorizing patients based on their laboratory test results to ensure timely and appropriate treatment.

- Optimize Resource Allocation: Accurately predict patient classifications to optimize resource allocation, including beds, staffing, and medical resources.


### Project Goal<a id="project-goal"></a>
The primary goal is to build a machine learning model using the provided dataset that accurately predicts whether a patient should be classified as in-care or out-care. The model should achieve a high level of accuracy and generalizability.


### Data Understanding<a id="data-understanding"></a>

#### Target and Features<a id="target-and-features"></a>
In this project, we have a dataset with the following target and features:

**Target Variable**
SOURCE: This is the target variable representing the patient treatment classification. It is a categorical variable with two classes: 1 for "in-care" and 0 for "out-care." The goal is to predict the patient's treatment classification based on the provided features.

**Features**
The dataset contains the following features:

- HAEMATOCRIT: Patient laboratory test result of haematocrit (continuous).
- HAEMOGLOBINS: Patient laboratory test result of haemoglobins (continuous).
- ERYTHROCYTE: Patient laboratory test result of erythrocyte (continuous).
- LEUCOCYTE: Patient laboratory test result of leucocyte (continuous).
- THROMBOCYTE: Patient laboratory test result of thrombocyte (continuous).
- MCH: Patient laboratory test result of MCH (continuous).
- MCHC: Patient laboratory test result of MCHC (continuous).
- MCV: Patient laboratory test result of MCV (continuous).
- AGE: Patient age (continuous).
- SEX: Patient gender (nominal - binary: "F" for female, "M" for male).
- SOURCE: The class target representing the patient treatment classification (nominal - binary: 1 for "in-care" and 0 for "out-care").


### Data Preparation<a id="">data-preparation</a>

Within our data preparation phase, we performed the following tasks:
 - Clean Data
    - Removed Duplicates
    - Filled Missing Values
    

### Explanatory Data Analysis<a id="">explanatory-data-analysis</a>

In this project, EDA is conducted to gain insights and understand the data in relation to the business problem, objectives, and goals.

The dataset contains both target and feature variables that are relevant to the project's objectives. The target variable is "SOURCE," which represents the patient treatment classification. It is a categorical variable with two classes: 1 for "in-care" and 0 for "out-care." The goal is to predict the patient's treatment classification based on the provided features.

During the EDA process, these variables are analyzed to gain a deeper understanding of their distributions, relationships, and potential impact on the patient treatment classification. By exploring histograms, stacked histograms, and the correlation matrix heatmap, we can uncover statistical insights and patterns that align with the business objectives of accurately classifying patients, optimizing resource allocation, and enhancing patient care.

### Modeling<a id="modelling"></a>
In the modeling phase, we employed various machine learning algorithms to develop predictive models for patient treatment classification based on the provided dataset. We explored a range of algorithms, including decision trees, KNN and random forests to identify the best-performing model.

To ensure reliable model performance, we conducted data preprocessing steps such as handling missing values, feature scaling, and encoding categorical variables. We split the dataset into training and testing sets, with a typical split of 80% for training and 20% for testing.

Next, we trained the models using the training data, fine-tuning their hyperparameters through techniques like grid search or random search. By optimizing the hyperparameters, we aimed to find the best configuration for each model that maximizes its predictive accuracy and generalization capabilities.

Additionally, we applied techniques such as feature selection or dimensionality reduction (e.g., Principal Component Analysis) to enhance model performance and alleviate issues related to multicollinearity or overfitting. These techniques aimed to identify the most informative features for patient treatment classification.

### Evaluation<a id="evaluation"></a>
During the evaluation phase, we rigorously assessed the performance of our trained models using various evaluation metrics and techniques. The primary evaluation metrics we employed include:

- Accuracy: The proportion of correct predictions out of the total number of predictions. It measures the overall correctness of the model's predictions.
- Precision: The proportion of true positive predictions out of the total positive predictions. It indicates the model's ability to accurately identify patients who require in-care treatment.
- Recall: The proportion of true positive predictions out of the total actual positive cases. It measures the model's ability to identify all patients who should be classified as in-care.
- F1-score: The harmonic mean of precision and recall. It provides a balanced evaluation metric that considers both precision and recall simultaneously.

In addition to these metrics, we examined the confusion matrix to gain deeper insights into the model's performance. The confusion matrix allowed us to analyze true positives, true negatives, false positives, and false negatives, providing a comprehensive understanding of the distribution of predictions.

To ensure the reliability and generalization of our models, we conducted cross-validation. Cross-validation involves splitting the dataset into multiple subsets and training/evaluating the model on different combinations of these subsets. This technique provides a more robust estimation of the model's performance by reducing the dependency on a single train-test split.

By comparing the performance of different models and techniques, we identified the most suitable model for our task based on its overall accuracy, precision, recall, F1-score, and robustness across cross-validation folds.

Overall, the modeling and evaluation phases were crucial in developing and selecting the most effective machine learning model for patient treatment classification. These phases ensured that the model's predictions are accurate, reliable, and aligned with the project's goals and objectives.

### Conclusion<a id="conclusion"></a>
Model Performance:

- KNN: The KNN model achieved an accuracy of 71%, precision of 64%, and recall of 63%.
- Decision Trees: The Decision Trees model achieved an accuracy of 69%, precision of 62%, and recall of 62%.
- Random Forest: The Random Forest model achieved the highest performance with an accuracy of 72%, precision of 65%, and recall of 63%. Additionally, the AUC score of 0.7888 indicates a good overall performance.

**Recommendations**
Based on these findings, I would recommend the Random Forest model as the preferred choice for the stakeholder. It exhibits the highest accuracy and precision among the evaluated models, indicating better overall predictive performance. Moreover, the Random Forest model's ability to capture feature importance provides valuable insights into the factors influencing the target variable.

In terms of predictive recommendations, it is important to consider the context and limitations of the model. The stakeholder should be aware that the model's predictions are based on the input variables used during training. Therefore, the model's predictions are expected to be most accurate when the input variables are similar to those encountered during training.







# Student Performance Prediction (STAT5009 Capstone Project)

## Overview

This project investigates the prediction of student performance in Portuguese language courses using supervised machine learning techniques. The objective is to identify students at risk of failing and support early educational intervention.

## Technologies Used

- R Programming Language
- RStudio IDE
- Supervised Machine Learning
- Logistic Regression
- LASSO Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Data Visualization and Statistical Analysis

## Dataset

The dataset contains academic, demographic, family, and social information for 649 Portuguese secondary school students.

Target Variable:

* **Pass** (G3 ≥ 10)
* **Fail** (G3 < 10)

Features include:

* Age, gender, and school
* Study time and previous failures
* Family and social characteristics
* School absences

## Methods

The following supervised learning models were implemented in R:

* Logistic Regression
* LASSO Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest

Model performance was evaluated using:

* Accuracy
* AUC
* Sensitivity
* Specificity

## Key Findings

* Random Forest achieved the highest overall accuracy (83.7%).
* KNN achieved the highest sensitivity (40%), making it more effective at identifying students at risk of failing.
* Previous failures and absences were among the most important predictors.

## Repository Contents

```text
├── student-por.csv                               # Portuguese student performance dataset
├── Group17.R                                     # Main R script for data analysis and modelling
├── Group 17 Project Presentation.pptx            # Project presentation slides
├── .Rhistory                                     # RStudio command history
├── README.md                                     # Project documentation
```


## Authors

Group 17

* Bhavana Thapa
* Rishabh Chhabra
* Laxmi Bhatta
* Syed Abrar Jamil
* Abu Fatah Mohammed Faisal

Curtin University

STAT5009 – Decision Methods and Predictive Analytics

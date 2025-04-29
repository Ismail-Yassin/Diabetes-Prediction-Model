# Project Title: **Diabetes Prediction Model**

Table of Contents

1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Significance](#significance)
4. [Frameworks](#frameworks)
5. [Dataset](#dataset-used)
6. [Methodology](#methodology)  
7. [Results](#results)
8. [Model_Limitations](#model-limitations)
9. [Future_improvements](#Future_improvements)
10.[Conclusion](#conclusion)

## Introduction

This project focuses on developing a machine learning model to predict diabetes using a dataset from the UCI Machine Learning Repository. The dataset contains various health-related features, and the goal is to build a predictive model that can assist in early diagnosis and intervention.

## Objectives

1. To perform Exploratory Data Analysis(EDA) to get insights about the data.
2. To prepare the data for Machine Learning.
3. To build a ML model
4. To deploy the model for use online

## Significance

- This model will help health care workers to detect the early signs of diabetes in patients.

- It will save a Doctor's time with diagnosis to determine which patients need immediate attention.

## Frameworks

The following are the tools that was used in this project.

- [Python](https://www.python.org) is a high level programming language that has got a lot of popularity in the data community and with the rapid growth of the libraries and frameworks, this is a right programming language to do ML.

- [NumPy](https://numpy.org) is a scientific computing tool used for array or matrix operations.

- [Pandas](https://pandas.pydata.org) is a great and simple tool for analyzing and manipulating data from a variety of different sources.

- [Matplotlib](https://matplotlib.org) is a comprehensive data visualization tool used to create static, animated, and interactive visualizations in Python.

- [Seaborn](https://seaborn.pydata.org) is another data visualization tool built on top of Matplotlib which is pretty simple to use.

- [Scikit-Learn](https://scikit-learn.org/stable/): Instead of building machine learning models from scratch, Scikit-Learn makes it easy to use classical models in a few lines of code.

- [Git & Github](https://github.com): Git is the framework that is used to track changes to the source file. Its majorly used for for version control. Github is where codes for projects get stored and colaborations can also worldwide from from the platform.

## Dataset used

This is the dataset used in this project: [Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

This dataset contains 4240 samples with 16 columns

The table below shows the first 10 samples of the data.

![Overview of the data](Assets\Data_overview.png)

## Methodology

**1. Dealing with missing values:**

This data contains some missing values in some of the columns. The table below shows the columns that have missing data, the amount of missing data and the percentage of missing values in the data

![missing data table](Assets\Missing_data.png)

The missing data was imputed using the **KnnImputer**. This method gets 4 data points near the missing data and then calculates their mean to impute the missing data. This avoid filling the missing data with a constant value.

**2. Duplicates:**

This dataset has no duplicated samples.

**3. Outlier detection:**

Thsi data has alot of missing values. Almost all the columns in contain outliers as seen in the image below;

![outlier data](Assets\boxplots.png)

The columns containing the most number of outliers are the **totChol** and **glucose** columns. The outliers in this data were not removed because the data is imbalanced and the minority class contained most of the outlier samples.

**5. The cleaning function used to preprocess the data:**

```python
# Cleaning function
def clean_data(filepath):
  '''
  This function cleans the data basing on the ideas gotten from EDA
  '''
  # Read the data
  df = pd.read_csv(filepath)

  # Dropping least correlated features ['male', 'prevalentStroke']
  df = df.drop(columns=['male', 'prevalentStroke'], axis=1)

  # Use knn imputer for filling mising values
  imputer = KNNImputer(n_neighbors=4)

  imputed = imputer.fit_transform(df)

  df = pd.DataFrame(imputed, columns=df.columns)

  # Changing 'diabetes' from float to int
  df['diabetes'] = df['diabetes'].apply(lambda x: 0 if x==0.0 else 1)

  return df

  ```

## Results

**1. Exploratory Data Analysis (EDA).**

This data is not equally distributed with the target vector with  the majority class having **97.4%** and the minority class having **2.6%** as shown by pie chart below;

The source code for distribution is below;

```python
distribution = data['diabetes'].value_counts(normalize=True)
plt.pie(x=distribution, autopct="%1.1f%%", labels=['Non-diabetic', 'Diabetic'], startangle=90)
plt.title("The distribution of patients")
plt.show()
```

![Distribution of the data](Assets\piechart_distribution_of_data.png)

There is no multicollinearity in the data. The threshold used was correlation coefficient of **0.8** and the correlation between the columns did not exceed the threshold value.

```python
threshold = 0.8
corr = data_EDA.drop('diabetes', axis=1).corr()>threshold
plt.subplots(figsize=(10, 5))
sns.heatmap(data=corr, annot=True)
plt.show()
```

![Heatmap](Assets\pearson_correlation.png)

The correlation of the feature columns with the target vector. The most correlated feature is **glucose** and the least correlated non negative number is **prevalentStroke**. The highest negative correlation is **currentSmoker**

The least correlated non negative features were removed from the dataset i.e. **prevalentStroke** and **male** columns

```python
corr_columns = data_EDA.drop("diabetes", axis=1).columns.to_list()
corr_data = {}

for col in corr_columns:
  correlation = data_EDA[col].corr(data_EDA['diabetes'])
  corr_data[col] = correlation

pd.Series(corr_data).sort_values(ascending=True).plot(kind='barh')
plt.xlim(-0.1, 0.7)
plt.title("Feature Vs Target correlation")
plt.xlabel("Correlation")
plt.show()

```

![Correlation with target vector](Assets\correlation_with_target_vector.png)

**2. Data preparation for Machine Learning.**

The data preprecessing stage was already explained in the methodologies including the source code for preprocessing the data for machine learning.

**3. Model development.**

In the model building process, Logistic Regression was used as the base line model and Decision tree was used as the iteration model. Based on the performance of the model, the logistics regression was used.

Choosing a model and also training the model the model

```python
# before performing the predictions using the model, I will first train it on a larger data
chosen_model = LogisticRegression(random_state=256)
chosen_model.fit(X_training_partition, y_training_partition)
```

The table below shows the performance of the model using metrics like accuracy score, precision, Recall, and F1 score

![model performance](Assets\model_performance.png)

The chart below shows the feature importance and the odds ratios used too predict each feature. For example;

- The **prevalentHyp** feature is the most important feature in this model with the odds ratio of **1.63**. This means that prevalentHyp levels is a main determinant for diabetes. In terms of using Odds ratios, for each unit of prevalentHyp increase, the odds of a patient having diabetes increase by 63%.

- The **TenYearOCH:** It is a decreasing feature to the predoction if diabetes in patients. For each unit decrease in the TenYearOCH, the odds of having diabetes decreases by 66%

- Odds ratios that are close or equal to 1 have no big effect on the target, odd ratios that are below 1 have decreasing effect on the target vector while the odds that are greater than 1 have an increasing chance of predicting the chance of having diabetes.

![Model performance with odds ratios](Assets\Model_performance_odds_ratios.png)

**4. Model deployment.**

The model was deployed using streamlit for interacting with the model.

## Model-Limitations

- The is imbalanced so the model might predict more of the dorminant class.

- The model has got outliers that were not removed because it contained most of the minority class. There it might end up predicting most of the minority class incase of positive cases for diabetes.

## Future_improvements

- Training the model using a balanced data to achieve a better accuracy.

- Using tree based algorithm like XGBoost or LightGBM to build the model to achieve a better accuracy.

## Conclusion

This project demonstrates how machine learning can be used to predict diabetes, providing valuable insights for early intervention and health management.

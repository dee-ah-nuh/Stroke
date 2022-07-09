**Dataset**

The dataset can be found in the [repository](https://github.com/dee-ah-nuh/stroke) or can be downloaded from [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

The dataset contains 5110 real world observations and 10 different features:

          gender: "Male", "Female" or "Other"
          age: age of the patient
          hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
          heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
          ever_married: "No" or "Yes"
          Residence_type: "Rural" or "Urban"
          avg_glucose_level: average glucose level in blood
          bmi: body mass index
          smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
          stroke: 1 if the patient had a stroke or 0 if not


**Features**

1. We employ an algorithm within the sytem that uses **data pre-processing** to handle object-type characters as well as null values.
2. The system uses a **70-30** training-testing split.
3. The system uses **Logistic Regression**, **Decision Trees** & **Bagging Classifier**

All these models respond to a variable (dependent variable) that has categorical values such as True/False or 0/1. It actually measures the probability of a binary response as the value of response variable based on the mathematical equation relating it with the predictor variables.
   
   
   
   
   
4. The system uses efficient and effective **visualization graphs** which help identify and understand important factors for stroke.

**Models**

- Logistic Regression
- Bagging Classifier

  **&**
- Decision Tree Classifier 

1. Input: The dataset
2. Output: Classification into 0 (no stroke) or 1 (stroke)


**Steps**
1. Loading the dataset and required libraries and packages

          import pandas as pd
          import numpy as np
          import matplotlib.pyplot as plt

          #Seaborn
          import seaborn as sns
          from seaborn import heatmap

          # Sci-kit learn
          from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\
              , classification_report, ConfusionMatrixDisplay
          from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
          from sklearn.compose import make_column_transformer, make_column_selector
          from sklearn.model_selection import train_test_split
          from sklearn.impute import SimpleImputer
          from sklearn.pipeline import make_pipeline

          #KMeans Clustering 

          from sklearn.cluster import KMeans
          from sklearn.metrics import silhouette_score
          from sklearn.linear_model import LogisticRegression
          from sklearn.model_selection import GridSearchCV
          from sklearn.metrics import accuracy_score, recall_score, precision_score, \
          f1_score, classification_report, confusion_matrix
          from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
          from sklearn import metrics
          from sklearn.ensemble import GradientBoostingClassifier
          
          
2. Pre-processing data to convert all characters to numeric and to remove NaN values
3. Dividing the dataset into training set and test set
4. Importing the Logistic Regression classifier, Decision Tree and Bagging Classifier and creating its object.
5. Fitting the training data to the classifier

          preprocessor.fit(X_train)
          ColumnTransformer(remainder='passthrough',
                  transformers=[('pipeline-1',
                                 Pipeline(steps=[('simpleimputer',
                                                  SimpleImputer()),
                                                 ('standardscaler',
                                                  StandardScaler())]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f4ceda8c4d0>),
                                ('pipeline-2',
                                 Pipeline(steps=[('simpleimputer',
                                                  SimpleImputer(fill_value='Missing',
                                                                strategy='constant')),
                                                 ('onehotencoder',
                                                  OneHotEncoder(handle_unknown='ignore',
                                                                sparse=False))]),
                                 ['gender', 'ever_married', 'work_type',
                                  'Residence_type', 'smoking_status'])])
                                  
6. Predicting the classifier output against the test data
7. Comparing the predicted results with the test results to get the accuracy

**KMeans Clustering**



**Key Findings**

- In this graph we can see the centroids and the various clusteres mapped upon age, bmi and average glucose level. We can see in the Elbow and the Intertia plot that the highest and optimal number of cluster to groups the people in tihs dataset was 3 as represented inn the scatterplot above.

![Figure 2022-07-07 203619 (22)](https://user-images.githubusercontent.com/96541076/178084360-01487939-ff6a-4e06-901b-08abcdc89bcf.png)


- This pie chart outlines the overall Stroke distribution based on the KMeans Clusters found above

![Figure 2022-07-07 203619 (5)](https://user-images.githubusercontent.com/96541076/178084784-2cb1fbd8-3797-43ff-be63-b637c224c9c6.png)
 
 
 - 
![Distribution of Float Variables](https://user-images.githubusercontent.com/96541076/178085196-0452dd3d-37d4-44df-8b3c-d339b0b5bb45.png)


- 
![difference between variables](https://user-images.githubusercontent.com/96541076/178085348-ce2a3f26-0bd5-494e-8add-4074c4b6dea2.png)


- 
![count plot for age_groups](https://user-images.githubusercontent.com/96541076/178085528-5d3a1257-baeb-4aa9-8810-02b3aaee7323.png)

**Developer**


-  Diana Valladares [github.com/dee-ah-nuh](https://github.com/dee-ah-nuh)

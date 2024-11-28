# Employee Attrition Prediction Neural Network

## Overview
In this project, I created a branched neural network that predicts employee attrition and identifies the most suitable department for each employee within a company. The model analyzes various employee features to generate two key predictions: whether an employee is likely to leave the company (attrition) and the department to which they would best fit.

## Background
As part of an initiative in Human Resources, I used machine learning to support workforce planning. By analyzing employee data, HR aims to predict potential attrition and better align employees with departments that match their skills and interests.

## Instructions
To preprocess the data and build the neural network model, I followed these steps:

### 1. Preprocessing the Data
```python
# Import dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

# Read employee attrition data
attrition_df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv')

# Target columns
y_df = attrition_df[['Attrition', 'Department']]

# Selected features for prediction
selected_columns = [
    'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'WorkLifeBalance'
]
X_df = attrition_df[selected_columns]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df)

# Scale X data
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# One-Hot Encoding the Department and Attrition
encoder_department = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
y_train_department = encoder_department.fit_transform(y_train[['Department']])
y_test_department = encoder_department.transform(y_test[['Department']])

encoder_attrition = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
y_train_attrition = encoder_attrition.fit_transform(y_train[['Attrition']])
y_test_attrition = encoder_attrition.transform(y_test[['Attrition']])
```

### 2. Building the Model
```python
from tensorflow.keras import layers, Model

# Define model architecture
num_features = len(X_train.columns)
inputs = layers.Input(shape=(num_features,), name='inputs')

# Shared Layers
shared_layer_1 = layers.Dense(64, activation='relu')(inputs)
shared_layer_2 = layers.Dense(128, activation='relu')(shared_layer_1)

# Department Branch
department_branch = layers.Dense(32, activation='relu')(shared_layer_2)
department_output = layers.Dense(y_train_department.shape[1], activation='sigmoid', name='department_output')(department_branch)

# Attrition Branch
attrition_branch = layers.Dense(32, activation='relu')(shared_layer_2)
attrition_output = layers.Dense(y_train_attrition.shape[1], activation='sigmoid', name='attrition_output')(attrition_branch)

# Create and compile model
model = Model(inputs=inputs, outputs=[department_output, attrition_output])
model.compile(optimizer='adam',
              loss={'department_output': 'categorical_crossentropy', 'attrition_output': 'binary_crossentropy'},
              metrics={'department_output': 'accuracy', 'attrition_output': 'accuracy'})
```

### 3. Training the Model
```python
# Train the model
model.fit(scaled_X_train, {'department_output': y_train_department, 'attrition_output': y_train_attrition}, epochs=100)
```

### 4. Evaluating the Model
```python
# Evaluate model
test_results = model.evaluate(scaled_X_test, {'department_output': y_test_department, 'attrition_output': y_test_attrition})
print(f'Department Accuracy: {test_results[2]}')
print(f'Attrition Accuracy: {test_results[1]}')
```

## Summary Questions
1. **Is accuracy the best metric to use on this data? Why or why not?**
    - Accuracy might not be the best metric due to potential class imbalance in the target categories, leading to misleading representations of model performance. Alternative metrics like precision, recall, F1-score, or AUC-ROC may provide better insights.

2. **What activation functions did you choose for your output layers, and why?**
    - I used the 'sigmoid' activation function for both output layers, as it is appropriate for multi-class binary classification, allowing the model to output a probability distribution across predicted classes.

3. **Can you name a few ways that this model could be improved?**
    - Possible improvements include modifying the architecture by adding hidden layers, tuning hyperparameters, experimenting with different activation functions, and addressing potential class imbalance.

## Sources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/documentation.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras) 


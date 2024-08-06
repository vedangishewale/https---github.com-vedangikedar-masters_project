#!/usr/bin/env python
# coding: utf-8

# Model 1

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[ ]:


# get_ipython().system('pip install statsmodels==0.14.2')


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Correct file path
file_path = './DSIB - Use Case Mortgage Holders - Data.csv'
df_mortgage= pd.read_csv(file_path)


# In[ ]:


df_mortgage


# In[ ]:


df_mortgage.describe()


# In[ ]:


df_mortgage.isnull().sum()


# In[ ]:


df_mortgage['Income'].describe()


# In[ ]:


num_columns= df_mortgage.select_dtypes(include=['float64', 'int64']).columns
num_correlation_matrix= df_mortgage[num_columns].corr()
plt.figure(figsize=(21, 14))
sns.heatmap(num_correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
plt.title('Correlation Heatmap of Mortgage Holders Data')
plt.show()


# In[ ]:


df_mortgage.columns


# In[ ]:


#Variance inflation factor (VIF) for verifying multi-correlinearity between features

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each numerical feature
X_vif = df_mortgage[num_columns].dropna()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

print(vif_data)


# In[ ]:


#Dropping features which have vif > 10 and 'inf'

X= X_vif.drop(columns=['ID', 'Has_Deposit', 'Not_Mortgage_Balance', 'not mortgage lending', 'deposit','InterestRate', 'TermInMonths'])
y_interest = X_vif['InterestRate']
y_term = X_vif['TermInMonths']


# In[ ]:


X


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
model=LinearRegression()
# Perform RFE for InterestRate
rfe_interest_rate = RFE(model, n_features_to_select=10)
fit_interest_rate = rfe_interest_rate.fit(X, y_interest)

# Perform RFE for TermInMonths
rfe_term_in_months = RFE(model, n_features_to_select=10)
fit_term_in_months = rfe_term_in_months.fit(X, y_term)

# Print selected features
print("Selected Features for InterestRate:", X.columns[fit_interest_rate.support_])
print("Selected Features for TermInMonths:", X.columns[fit_term_in_months.support_])


# In[ ]:


X_interest = X[X.columns[fit_interest_rate.support_]]
interest_model= model.fit(X_interest,y_interest)
importance_interest = interest_model.coef_
# summarize feature importance
for i,v in enumerate(importance_interest):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance_interest))], importance_interest)
plt.show()

# Fit the model for TermInMonths
# term_model= model.fit(X, y_term)
# feature_importances_term_in_months = term_model.feature_importances_


# # Linear regression for Predicting Interest Rate without Principal component analysis(PCA)

# In[ ]:


X_interest_train, X_interest_test, y_interest_train, y_interest_test = train_test_split(X_interest, y_interest, test_size=0.2, random_state=42)
model.fit(X_interest_train, y_interest_train)


# In[ ]:


y_pred_interest = model.predict(X_interest_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_interest_test, y_pred_interest)
r2 = r2_score(y_interest_test, y_pred_interest)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_interest_test, y_pred_interest, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_interest_test.min(), y_interest_test.max()], [y_interest_test.min(), y_interest_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Interest Rate')
plt.show()


# # Linear regression for Predicting Interest Rate with Principal component analysis(PCA)

# In[ ]:


# performed PCA and explained how does variance vary with number of components


features = ['Beacon_Score', 'Services', 'Avg_Monthly_Transactions', 'Has_Payroll',
            'Has_Investment', 'Has_Visa', 'Age', 'Tenure_In_Months',
            'TermToMaturity', 'NumberOfParties']

X = df_mortgage[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=len(features))  # Set the number of components to the number of features
X_pca = pca.fit_transform(X_scaled)

# Plot the explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid()
plt.show()

# Create a DataFrame with the PCA components
pca_components = pd.DataFrame(pca.components_, columns=features)
explained_variance = pca.explained_variance_ratio_

print("Explained Variance Ratio:\n", explained_variance.cumsum())
print("PCA Components:\n", pca_components)


# In[ ]:


features = ['Beacon_Score', 'Services', 'Avg_Monthly_Transactions', 'Has_Payroll',
            'Has_Investment', 'Has_Visa', 'Age', 'Tenure_In_Months',
            'TermToMaturity', 'NumberOfParties']
target = 'InterestRate'

X = df_mortgage[features]
y = df_mortgage[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=len(features))  # Set the number of components to the number of features
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize the model
model1 = LinearRegression()

# Train the model
model1.fit(X_train, y_train)

# Predict on the test set
y_pred = model1.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Interest Rate (PCA + Linear Regression)')
plt.show()


# In[ ]:


import pickle
pickle.dump(model1, open('linear_regression_model.pkl', 'wb'))


# # Support Vector Regression for predicting Interest Rate

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_interest_train)

# Transform the test data
X_test_scaled = scaler.transform(X_interest_test)

# Initialize the SVR model
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Train the model
svr.fit(X_train_scaled, y_interest_train)

# Predict on the test set
y_svr_interest_pred = svr.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_interest_test, y_svr_interest_pred)
r2 = r2_score(y_interest_test, y_svr_interest_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_interest_test, y_svr_interest_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_interest_test.min(), y_interest_test.max()], [y_interest_test.min(), y_interest_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Interest Rate (SVR)')
plt.show()


# # Support Vector regression for Predicting Interest Rate with Principal component analysis(PCA)

# In[ ]:


# Standardize the features
svr_scaler = StandardScaler()
X_scaled_svr = scaler.fit_transform(X)

# Perform PCA
svr_pca = PCA(n_components=len(features))  # Set the number of components to the number of features
X_pca_svr = pca.fit_transform(X_scaled_svr)

# Split the data into training and testing sets
X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(X_pca_svr, y, test_size=0.2, random_state=42)

# Initialize the model
svr_pca_model = SVR(kernel='rbf')  # You can also try 'linear' or 'poly' kernels

# Train the model
svr_pca_model.fit(X_train_svr, y_train_svr)

# Predict on the test set
y_pred_svr_pca = model.predict(X_test_svr)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_svr, y_pred_svr_pca)
r2 = r2_score(y_test_svr, y_pred_svr_pca)

# Calculate adjusted R^2
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
print(f"Adjusted R^2 Score: {adjusted_r2}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_svr, y_pred_svr_pca, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test_svr.min(), y_test_svr.max()], [y_test_svr.min(), y_test_svr.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Interest Rate (PCA + SVR)')
plt.show()


# <hr>

# ## Dataset 2

# ### Data Cleaning

# In[ ]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[ ]:


#Load the dataset
file_path = '/work/Part2.csv'  
df2 = pd.read_csv(file_path)


# In[ ]:


# Display the first few rows of the dataframe
print(df2.head(5))


# ## Data Cleaning

# In[ ]:


# Filling missing numerical values with the mean of the column
numerical_cols = ['Beacon_Score', 'Mortgage_Balance', 'Avg_Monthly_Transactions', 'VISA_balance',
                  'Not_Mortgage_Balance', 'Services', 'Tenure_In_Months', 'TermInMonths',
                  'TermToMaturity', 'InterestRate']

for col in numerical_cols:
    df2[col].fillna(df2[col].mean(), inplace=True)


# In[ ]:


# Checking for missing values
print(df2.isnull().sum())


# In[ ]:


#Label Encoding where 0 is renewed and 1 is run off
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
le = LabelEncoder()

# Fit and transform the Closing_status column
df2['Closing_status'] = le.fit_transform(df2['Closing_status'])

# Verify the encoding
print(df2['Closing_status'].unique())


# ### EDA

# In[ ]:


# Summary statistics
print(df2.describe())


# In[ ]:


# Correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


# Distribution plots
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df2[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[ ]:


# Box plots for categorical variables
categorical_cols = ['Has_Payroll', 'Has_Investment', 'Has_Visa', 'Has_Deposit', 'Closing_status']
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=col, y='Beacon_Score', data=df2)
    plt.title(f'Beacon_Score by {col}')
    plt.show()


# In[ ]:


# Feature Selection
# Dropping 'ID' as it's not relevant for clustering
df2.drop(columns=['ID'], inplace=True)


# In[ ]:


# Prepare the data for clustering
features = ['Beacon_Score', 'Mortgage_Balance', 'Avg_Monthly_Transactions', 'Has_Payroll',
            'Has_Investment', 'Has_Visa', 'VISA_balance', 'Has_Deposit', 'Not_Mortgage_Balance',
            'Services', 'Tenure_In_Months', 'TermInMonths', 'TermToMaturity', 'InterestRate']
X = df2[features]


# In[ ]:


# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ### Customer Segmentation

# K-means clustering to segment customers

# In[ ]:


# Determine the optimal number of clusters using the elbow method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# In[ ]:


# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()


# In[ ]:


# From the elbow plot, choose the optimal number of clusters (k=3)
optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


# In[ ]:


# Add the cluster labels to the original dataframe
df2['Cluster'] = clusters


# In[ ]:


# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df2['PCA1'] = X_pca[:, 0]
df2['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df2, palette='viridis')
plt.title('Customer Segments')
plt.show()


# In[ ]:


# Analyze the characteristics of each cluster
cluster_summary = df2.groupby('Cluster').mean()
print(cluster_summary)


# In[ ]:


# Further analysis on Closing_status within each cluster
closing_status_summary = df2.groupby(['Cluster', 'Closing_status']).size().unstack(fill_value=0)
print(closing_status_summary)


# <hr>

# In[ ]:


# Adding a column for cluster labels based on the analysis
cluster_labels = {
    0: "Low Activity, High Credit Score",
    1: "Moderate Activity, High Mortgage",
    2: "High Activity, High Mortgage",
    3: "Moderate Activity, Moderate Mortgage",
    4: "High Activity, Very High Mortgage",
    5: "High Activity, Low Mortgage",
    6: "Low Activity, Low Credit Score"
}

df2['Cluster_Label'] = df2['Cluster'].map(cluster_labels)

# Display the dataframe with cluster labels
print(df2[['Cluster', 'Cluster_Label']].head())


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=65944878-a93b-4d2b-98b4-f100dda5388f' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

#!/usr/bin/env python
# coding: utf-8

# # Height Relationship Prediction Analysis

# - Nama: Bryan Herdianto
# - Email: bryan.herdianto17@gmail.com
# - ID Dicoding: bryanherdianto

# ## Persiapan

# ### Menyiapkan library yang dibutuhkan

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.preprocessing import LabelEncoder


# ### Menyiapkan data yang digunakan

# In[2]:


# Load dataset
data = pd.read_csv('https://raw.githubusercontent.com/bryanherdianto/dicoding-indonesia/main/height-relationship/GaltonFamilies.csv')


# In[3]:


data.head(10)


# In[4]:


data.info()


# In[5]:


data.describe()


# ## Data Cleaning

# In[6]:


data.drop('rownames', axis=1, inplace=True)


# ## Data Understanding

# In[7]:


# Distribution of childHeight
plt.style.use('dark_background')
sns.set_style('darkgrid')

plt.figure(figsize=(8, 6))
sns.histplot(data['childHeight'], bins=20, kde=True, color='blue', edgecolor='skyblue')
plt.title('Distribution of Child Height')
plt.xlabel('Child Height (inches)')
plt.ylabel('Frequency')
plt.show()


# In[8]:


# Distribution of father's height
plt.style.use('dark_background')
sns.set_style('darkgrid')

plt.figure(figsize=(8, 6))
sns.histplot(data['father'], bins=20, kde=True, color='orange', edgecolor='bisque')
plt.title("Distribution of Father's Height")
plt.xlabel('Father Height (inches)')
plt.ylabel('Frequency')
plt.show()


# In[9]:


# Distribution of mother's height
plt.style.use('dark_background')
sns.set_style('darkgrid')

plt.figure(figsize=(8, 6))
sns.histplot(data['mother'], bins=20, kde=True, color='red', edgecolor='salmon')
plt.title("Distribution of Mother's Height")
plt.xlabel('Mother Height (inches)')
plt.ylabel('Frequency')
plt.show()


# In[10]:


# Pair plot height of family members
plt.style.use('dark_background')
sns.set_style('darkgrid')

palette = {'male': 'skyblue', 'female': 'red'}
sns.pairplot(data[['father', 'mother', 'midparentHeight', 'childHeight', 'gender']], hue='gender', palette=palette)
plt.suptitle('Pair Plot of Parent and Child Heights', y=1.02, color='white')
plt.show()


# In[11]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='gender', y='childHeight', data=data, inner='quartile', palette=palette, hue='gender')
plt.title('Violin Plot of Child Height by Gender')
plt.xlabel('Gender')
plt.ylabel('Child Height (inches)')
plt.show()


# In[12]:


correlation_matrix = data[['father', 'mother', 'midparentHeight', 'childHeight']].corr()
colors = ["salmon", "red"]
n_bins = 100  
cmap_name = 'blue_orange'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.set(style='darkgrid')
sns.heatmap(correlation_matrix, annot=True, cmap=cm, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix with Annotations', color='white')
plt.show()


# ## Data Preparation / Preprocessing

# ### Label Encoder

# In[13]:


# Label encode 'family' and 'gender' columns
label_encoder_family = LabelEncoder()
label_encoder_gender = LabelEncoder()

data['family'] = label_encoder_family.fit_transform(data['family'])
data['gender'] = label_encoder_gender.fit_transform(data['gender'])


# In[14]:


data.head()


# ### Split training and test data

# In[15]:


# Select relevant columns for modeling
X = data[['father', 'mother', 'midparentHeight', 'family', 'gender']]
y = data['childHeight']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=12)


# ### Hyperparameter tuning

# In[16]:


# Define model Random Forest Regressor
model_rf = RandomForestRegressor(random_state=1)

# Define parameter grid
param_grid_rf = {
    'n_estimators': randint(50, 200), 
    'max_features': ['sqrt', 'log2'],  
    'max_depth': randint(5, 20), 
    'min_samples_split': randint(2, 10),  
    'min_samples_leaf': randint(1, 5)  
}

# Perform RandomizedSearchCV
random_search_rf = RandomizedSearchCV(estimator=model_rf, param_distributions=param_grid_rf, n_iter=100,
                                      scoring='neg_mean_squared_error', cv=5, random_state=1)
random_search_rf.fit(X_train, y_train)


# In[18]:


# Define model Gradient Boosting Regressor
model_gb = GradientBoostingRegressor(random_state=1)

# Define parameter grid
param_grid_gb = {
    'n_estimators': randint(50, 200), 
    'learning_rate': uniform(0.01, 0.3), 
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'subsample': uniform(0.5, 0.5)
}

# Perform RandomizedSearchCV
random_search_gb = RandomizedSearchCV(estimator=model_gb, param_distributions=param_grid_gb, n_iter=100,
                                      scoring='neg_mean_squared_error', cv=5, random_state=1)
random_search_gb.fit(X_train, y_train)


# In[19]:


# Define model AdaBoost Regressor
model_ab = AdaBoostRegressor(random_state=1)

# Define parameter grid
param_grid_ab = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 1.0),
    'loss': ['linear', 'square', 'exponential']
}

# Perform RandomizedSearchCV
random_search_ab = RandomizedSearchCV(estimator=model_ab, param_distributions=param_grid_ab, n_iter=100,
                                      scoring='neg_mean_squared_error', cv=5, random_state=1)
random_search_ab.fit(X_train, y_train)


# ## Modeling

# In[20]:


# Initialize Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Calculate metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression")
print(f'MSE: {mse_lr}')
print(f'R2 Score: {r2_lr}')


# In[21]:


# Get best model of Random Forest Regressor from hyperparameter tuning
best_model_rf = random_search_rf.best_estimator_
y_pred_rf = best_model_rf.predict(X_test)

# Calculate metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest Regressor with Hyperparameter Tuning")
print("Best Parameters:", random_search_rf.best_params_)
print(f'MSE: {mse_rf}')
print(f'R2 Score: {r2_rf}')


# In[28]:


# Initialize Multi-Layer Perceptron Regressor
model_mlp = MLPRegressor(random_state=1, max_iter=500)
model_mlp.fit(X_train, y_train)
y_pred_mlp = model_mlp.predict(X_test)

# Calculate metrics
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
print("Multi-Layer Perceptron (Neural Network)")
print(f'MSE: {mse_mlp}')
print(f'R2 Score: {r2_mlp}')


# In[23]:


# Get best model of Gradient Boosting Regressor from hyperparameter tuning
best_model_gb = random_search_gb.best_estimator_
y_pred_gb = best_model_gb.predict(X_test)

# Calculate metrics
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print("Gradient Boosting Regressor")
print("Best Parameters:", random_search_gb.best_params_)
print(f'MSE: {mse_gb}')
print(f'R2 Score: {r2_gb}')


# In[24]:


# Get best model of AdaBoost Regressor from hyperparameter tuning
best_model_ab = random_search_ab.best_estimator_
y_pred_ab = best_model_ab.predict(X_test)

# Calculate metrics
mse_ab = mean_squared_error(y_test, y_pred_ab)
r2_ab = r2_score(y_test, y_pred_ab)
print("AdaBoost Regressor")
print("Best Parameters:", random_search_ab.best_params_)
print(f'MSE: {mse_ab}')
print(f'R2 Score: {r2_ab}')


# ## Evaluation

# In[29]:


# Compare Model Performances
models = [
    'Linear Regression', 
    'Random Forest Regressor with Hyperparameter Tuning', 
    'MLP Regressor', 
    'Gradient Boosting Regressor with Hyperparameter Tuning', 
    'AdaBoost Regressor with Hyperparameter Tuning'
]
mse_scores = [mse_lr, mse_rf,  mse_mlp, mse_gb, mse_ab]
r2_scores = [r2_lr, r2_rf, r2_mlp, r2_gb, r2_ab]

performance_df = pd.DataFrame({
    'Model': models,
    'MSE': mse_scores,
    'R2 Score': r2_scores
})

print("Model Performance Comparison:")
performance_df


# In[42]:


print(f"Best model: {performance_df.loc[performance_df['MSE'].idxmin(), 'Model']}")


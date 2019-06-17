
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import glob

os.chdir('C:/Users/natas/Desktop/safety/safety/features/')
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "df.csv", index=False, encoding='utf-8-sig')


# In[2]:


import pandas as pd
import os
import glob


# In[3]:


df_label = pd.read_csv("C:/Users/natas/Desktop/safety/safety/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")


# In[4]:


df= pd.read_csv('C:/Users/natas/Desktop/safety/safety/features/df.csv')


# In[5]:


df = pd.merge(df, df_label, on='bookingID')


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


import numpy as np
cols = ['bookingID','Accuracy','Bearing', 'acceleration_x', 'acceleration_y', 'acceleration_z',
'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']
df[cols] = df[cols].astype(np.int64)


# In[9]:


df.head()


# In[10]:


#Logic- Dangerous driving involves looking at acceleration and bearing data. If there is a high acceleration and high change in 
#bearing then you have a case of sharp turns at high speeds--> Dangerous

 

df['accelerationx2']= ((df['acceleration_x']**2) + (df['acceleration_y']**2) + (df['acceleration_z']**2))

df['resultant_acc'] = df['accelerationx2']**0.5 

df['MA_acc'] = df.rolling(window=2)['resultant_acc'].mean()

df['MA_acc_max']= df.groupby('bookingID')['MA_acc'].transform('max')
df['MA_acc_mean']= df.groupby('bookingID')['MA_acc'].transform('mean')
df['MA_acc_skew']= abs(df.groupby('bookingID')['MA_acc'].transform('skew'))
df['MA_acc_var']= df.groupby('bookingID')['MA_acc'].transform('var')


#Check if speed > speed limit. Max speed limit in Singapore is 80-90 km/h ` 25m/s
df['Speedmax'] = df.groupby('bookingID')['Speed'].transform('max')
df['Speedabovelimit'] = df['Speedmax'].apply(lambda x: 1 if x >25 else 0)

#change in bearing

df['bearing_chg'] = df.rolling(window=2)['Bearing'].mean() #assumption all bearings are vs true north measure clockwise"
df['bearing_chg_max']= df.groupby('bookingID')['bearing_chg'].transform('max')
df['bearing_chg_var']= df.groupby('bookingID')['bearing_chg'].transform('var')


#combination of acceleration and bearing - rudimenatary feature creation. looking for high acceleration and high bearing
df['bearing_acc_comb1'] = df['MA_acc_max'] +  df['bearing_chg_max']
df['bearing_acc_comb2'] = df['MA_acc_var'] +  df['bearing_chg_var']


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df = df.drop_duplicates(subset='bookingID', keep='last')


# In[ ]:


df.head()


# In[ ]:


df.isnull().values.any() #there shouldnt be any null values


# In[ ]:


from sklearn.preprocessing import label_binarize

df['label']= label_binarize(df['label'], classes=[0,1])


# In[ ]:


df.shape


# In[ ]:


df = df.drop(['bookingID','Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y',
       'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed','Speedmax','accelerationx2','MA_acc','resultant_acc'], axis=1)


# In[ ]:


df = df.reindex(np.random.permutation(df.index))


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


import seaborn as sns
import numpy as np

corr = df.corr()
sns.heatmap(corr)


# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np 
labels = df.label
features = df.drop('label', axis=1)


# In[ ]:


# split data into train and test sets
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=seed)


# In[ ]:


print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Labels Shape:', y_test.shape)


# In[ ]:


from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)


# In[ ]:


np.bincount(y_train)


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000)
# Train the model on training data
rf.fit(X_train, y_train);


# In[ ]:


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
y_pred_class = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]
y_pred_prob
metrics.roc_auc_score(y_test, y_pred_class)


# In[ ]:


from pprint import pprint
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[ ]:


#Hyperparameter tuning for random forest
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, limited due to computational issues
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[ ]:


rf_random.best_params_ #output: {'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': False}


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rfhyper = RandomForestClassifier(n_estimators = 1400, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth = 40, bootstrap = False)
# Train the model on training data
rfhyper.fit(X_train, y_train);


# In[ ]:


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
y_pred_class = rfhyper.predict(X_test)
y_pred_prob = rfhyper.predict_proba(X_test)[:, 1]
y_pred_prob
metrics.roc_auc_score(y_test, y_pred_class)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [ 0.00001, 0.0001, 0.001, 0.01,0.1, 1, 10, 100]}
LRmodel = RandomizedSearchCV(LogisticRegression(), param_grid, cv=3, scoring='roc_auc')

LRmodel.fit(X_train, y_train)

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
y_pred_class = LRmodel.predict(X_test)
y_pred_prob = LRmodel.predict_proba(X_test)[:, 1]
y_pred_prob
metrics.roc_auc_score(y_test, y_pred_class)


# In[ ]:


from pprint import pprint
print('Parameters currently in use:\n')
pprint(model.get_params())


# In[ ]:


import xgboost as xgb


# In[ ]:


# split data into train and test sets
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=seed)


# In[ ]:


xg_class = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


# In[ ]:


#XGBoost base model
from numpy import loadtxt
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
y_pred_class = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_prob
metrics.roc_auc_score(y_test, y_pred_class)


# In[ ]:


#Hyperparameters for XGBoost
random_grid ={
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[ ]:


xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='reg:logistic',
                    silent=True, nthread=1)


# In[ ]:


#Hyperparameter tuning for XGBoost
from sklearn.model_selection import RandomizedSearchCV
from numpy import loadtxt
from xgboost import XGBClassifier

random_search = RandomizedSearchCV(xgb, param_distributions=random_grid, n_iter=5, scoring='roc_auc', n_jobs=4, cv=3, verbose=3, random_state=1001 )
random_search.fit(X_train, y_train)


# In[ ]:


print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (3, 5))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)


# In[ ]:


modeltuned = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=2, learning_rate=0.02, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=600,
       n_jobs=1, nthread=1, objective='reg:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.6)
modeltuned.fit(X_train, y_train)


# In[ ]:


y_pred_class = modeltuned.predict(X_test)
y_pred_prob = modeltuned.predict_proba(X_test)[:, 1]
y_pred_prob
metrics.roc_auc_score(y_test, y_pred_class)


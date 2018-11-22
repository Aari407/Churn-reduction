
# coding: utf-8

# In[1]:


import os
os.chdir("D:/Project")
os.getcwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve


# In[2]:


df_train=pd.read_csv('Train_data.csv', sep=',')
df_train.head()
df_train.info()
assert df_train.notnull().all().all()
df_train['phone number']=df_train['phone number'].apply(lambda x: x.replace('-', ''))
df_train['phone number']=pd.to_numeric(df_train['phone number'], errors='coerce')
cat_list=['international plan', 'voice mail plan', 'Churn', 'state']
categorize_label=lambda x:x.astype('category')
df_train[cat_list]=df_train[cat_list].apply(categorize_label, axis=0)
df_train.info()
df_train_pd=pd.get_dummies(df_train[cat_list], drop_first=True, prefix_sep='-')
df_train_cat=df_train.drop(cat_list, axis=1).join(df_train_pd)
df_train_cat.info()
target_train = df_train_cat['Churn- True.']
features_train = df_train_cat.drop('Churn- True.',axis=1)



# In[3]:


df_test=pd.read_csv('Test_data.csv', sep=',')
assert df_test.notnull().all().all()
df_test['phone number']=df_test['phone number'].apply(lambda x: x.replace('-', ''))
df_test['phone number']=pd.to_numeric(df_train['phone number'], errors='coerce')
df_test[cat_list]=df_test[cat_list].apply(categorize_label, axis=0)
df_test_pd=pd.get_dummies(df_test[cat_list], drop_first=True, prefix_sep='-')
df_test_cat=df_test.drop(cat_list, axis=1).join(df_test_pd)
target_test = df_test_cat['Churn- True.']
features_test = df_test_cat.drop('Churn- True.',axis=1)


# In[25]:


n_churn = len(df_train_cat)
print(df_train_cat['Churn- True.'].value_counts())
print(df_train_cat['Churn- True.'].value_counts()/n_churn*100)
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix)


# In[23]:


""""
model = DecisionTreeClassifier(random_state=42)
model.fit(features_train,target_train)
prediction_dtc=model.predict(features_test)
print(model.score(features_train,target_train)*100)
print(model.score(features_test,target_test)*100)
print(recall_score(target_test, prediction_dtc) * 100)
print(roc_auc_score(target_test, prediction_dtc) * 100)
""""



# In[15]:


""""
feature_importances = model.feature_importances_
feature_list = list(features_train)
relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])
relative_importances.sort_values(by="importance", ascending=False)
selected_features = relative_importances[relative_importances.importance>0.01]
selected_list = selected_features.index
features_train_selected = features_train[selected_list]
features_test_selected = features_test[selected_list]
relative_importances.plot(kind='barh')
plt.title('Features Importances')
plt.show()
""""


# In[9]:


""""
depth = [i for i in range(5,21,1)]
samples = [i for i in range(50,500,50)]
parameters = dict(max_depth=depth, min_samples_leaf=samples)
param_search = GridSearchCV(model, parameters)
param_search.fit(features_train, target_train)
print(param_search.best_params_)
""""


# In[13]:


""""
model_best = DecisionTreeClassifier(max_depth = 5,  min_samples_leaf = 50,  class_weight = "balanced", random_state=42)
model_best.fit(features_train_selected, target_train)
prediction_best = model_best.predict(features_test_selected)
print(model_best.score(features_test_selected, target_test) * 100)
print(recall_score(target_test, prediction_best) * 100)
print(roc_auc_score(target_test, prediction_best) * 100)
""""


# In[22]:


""""
fpr, tpr, _ = roc_curve(target_test, prediction_best)
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve of Descion Tree')
plt.show()
""""


# In[ ]:


model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(features_train,target_train)
prediction_best_rf=model_rf.predict(features_test)
model_rf.score(features_train,target_train)*100
print(model_rf.score(features_test,target_test)*100)
print(recall_score(target_test, prediction_best_rf) * 100)
print(roc_auc_score(target_test, prediction_best_rf) * 100)


# In[ ]:


feature_importances_rf = model_rf.feature_importances_
feature_list_rf = list(features_train)
relative_importances_rf = pd.DataFrame(index=feature_list_rf, data=feature_importances_rf, columns=["importance"])
relative_importances_rf.sort_values(by="importance", ascending=False)
selected_features_rf = relative_importances_rf[relative_importances_rf.importance>0.01]
selected_list_rf = selected_features_rf.index
features_train_selected_rf = features_train[selected_list_rf]
features_test_selected_rf = features_test[selected_list_rf]
relative_importances_rf.plot(kind='barh', rot=45)
ax = plt.axes()
ax.legend_.remove()
plt.title('Features Importances')
plt.show()


# In[ ]:


model_best_rf = RandomForestClassifier(n_estimators=1000,
            random_state=2)
model_best_rf.fit(features_train_selected_rf, target_train)
prediction_best_rf = model_best_rf.predict(features_test_selected_rf)
print(recall_score(target_test, prediction_best_rf) * 100)
print(roc_auc_score(target_test, prediction_best_rf) * 100)
print(accuracy_score(target_test, prediction_best_rf) * 100)


# In[ ]:


params_rf = {'n_estimators':[i for i in range(500, 1100, 100)], 'random_state':[42], 'min_samples_leaf':[i for i in range(10,60,10)], 'class_weight':["balanced"], 'max_depth': [i for i in range(10,60,10)]}
grid_rf = GridSearchCV(estimator=model_best_rf,
                       param_grid=params_rf,
                       n_jobs=-1)
grid_rf.fit(features_train_selected_rf, target_train)
best_model = grid_rf.best_estimator_

y_pred_rf = best_model.predict(features_test_selected_rf)
print(roc_auc_score(target_test, y_pred_rf) * 100)
print(recall_score(target_test, y_pred_rf) * 100)
print(accuracy_score(target_test, y_pred_rf) * 100)


# In[ ]:


fpr, tpr, _ = roc_curve(target_test, y_pred_rf)
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve of Random Forest')
plt.show()


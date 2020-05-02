#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary packages: 
import pandas as pd 
import random as rand
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.linear_model


# In[2]:


# Extracting and reading file: 
file = 'app_chef_model_version.xlsx'
app_chef = pd.read_excel(file)


# In[3]:


# Preparing data for statsmodel: 

# Dropping revenue from the explanatory variable set:
app_chef_explanatory = app_chef.drop(['revenue', 'share_revenue',
                                     'out_revenue_hi','out_share_revenue_hi'], axis = 1)

# Dropping also discrete variables: 
app_chef_explanatory = app_chef_explanatory.drop(['name','email','first_name','family_name',
                                                  'domain', 'domain_group', 'rating_category',
                                                 'cust_serv_status'],axis = 1)

# Formatting explanatory variable for statsmodels:
for val in app_chef_explanatory:
    print(f"app_chef['{val}'] +")


# In[4]:


# Building statsmodel with all explanatory variables: 

lm_full = smf.ols(formula ="""  app_chef['revenue']~
                                app_chef['cross_sell'] +
                                app_chef['total_meals'] +
                                app_chef['unique_meals'] +
                                app_chef['contact_cust_serv'] +
                                app_chef['prod_cat_view'] +
                                app_chef['avg_time_vis'] +
                                app_chef['mob_number'] +
                                app_chef['early_cancel'] +
                                app_chef['late_cancel'] +
                                app_chef['tastes_pref'] +
                                app_chef['pc_logins'] +
                                app_chef['mob_logins'] +
                                app_chef['week_plan'] +
                                app_chef['early_deliv'] +
                                app_chef['late_deliv'] +
                                app_chef['pack_lock'] +
                                app_chef['ref_lock'] +
                                app_chef['follow_rec_pct'] +
                                app_chef['avg_prep_time'] +
                                app_chef['larger_order'] +
                                app_chef['master_classes'] +
                                app_chef['med_meal_rate'] +
                                app_chef['avg_clicks'] +
                                app_chef['total_photos'] +
                                app_chef['m_family_name'] +
                                app_chef['avg_tckt_order'] +
                                app_chef['avg_contact_cust_serv'] +
                                app_chef['pct_late_deliv'] +
                                app_chef['pct_early_deliv'] +
                                app_chef['share_total_meals'] +
                                app_chef['out_avg_time_hi'] +
                                app_chef['out_avg_prep_lo'] +
                                app_chef['out_avg_prep_hi'] +
                                app_chef['out_total_meals_hi'] +
                                app_chef['out_unique_meals_hi'] +
                                app_chef['out_cont_cust_serv_lo'] +
                                app_chef['out_cont_cust_serv_hi'] +
                                app_chef['out_canc_bef_noon_hi'] +
                                app_chef['out_late_deliv_hi'] +
                                app_chef['out_larg_order_lo'] +
                                app_chef['out_larg_order_hi'] +
                                app_chef['out_avg_clicks_lo'] +
                                app_chef['out_avg_clicks_hi'] +
                                app_chef['out_total_photos_hi'] +
                                app_chef['out_pct_late_deliv_hi'] +
                                app_chef['out_pct_early_deliv_hi'] +
                                app_chef['out_share_total_meals_hi'] +
                                app_chef['thr_total_meals'] +
                                app_chef['thr_unique_meals'] +
                                app_chef['thr_contact_cust_serv'] +
                                app_chef['thr_avg_time_vis'] +
                                app_chef['thr_early_cancel'] +
                                app_chef['thr_late_cancel'] +
                                app_chef['thr_late_deliv'] +
                                app_chef['thr_avg_prep_time'] +
                                app_chef['thr_total_photos'] +
                                app_chef['thr_avg_tckt_order'] +
                                app_chef['thr_pct_late_deliv'] +
                                app_chef['thr_pct_early_deliv'] +
                                app_chef['Negative'] +
                                app_chef['Neutral'] +
                                app_chef['Positive'] +
                                app_chef['business'] +
                                app_chef['personal'] +
                                app_chef['Satisfied'] +
                                app_chef['Unhappy']""", 
                  data = app_chef)

# Fitting the results:
results_full = lm_full.fit()

# Printing summary:
results_full.summary()


# In[5]:


# Selecting explanatory variables and splitting dataset: 

# Dropping all insignificant variables:
app_chef_data   = app_chef_explanatory.drop(['tastes_pref','mob_logins','med_meal_rate','week_plan',
                                             'mob_number','mob_number','pack_lock','ref_lock','pc_logins',
                                             'prod_cat_view','early_cancel','late_cancel','early_deliv',
                                             'follow_rec_pct','pct_early_deliv','avg_clicks','out_late_deliv_hi',
                                             'out_larg_order_lo','out_larg_order_hi','out_avg_clicks_lo',
                                             'out_avg_clicks_hi','out_total_photos_hi','thr_unique_meals',
                                             'thr_early_cancel','thr_late_deliv','thr_avg_prep_time',
                                             'thr_pct_early_deliv', 'out_pct_late_deliv_hi', 
                                             'out_pct_early_deliv_hi','share_total_meals'],
                                    axis = 1)

# Selecting response variable:
app_chef_target = app_chef.loc[:, 'revenue']


# Splitting dataset:
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size = 0.25,
            random_state = 222)

# Checking training set:
print(X_train.shape)
print(y_train.shape)

# Checking testing set:
print(X_test.shape)
print(y_test.shape)

# Selecting explanatory variables:
x_variables = ['total_meals', 'unique_meals', 'contact_cust_serv', 'avg_time_vis','late_deliv','avg_prep_time',
               'larger_order', 'master_classes','total_photos','thr_total_meals','thr_contact_cust_serv',
               'thr_avg_time_vis','thr_late_cancel','thr_total_photos','Negative','Neutral','Positive',
               'business','personal', 'avg_tckt_order','avg_contact_cust_serv',
               'pct_late_deliv','thr_avg_tckt_order','thr_pct_late_deliv', 'Satisfied', 'Unhappy']

# Creating loop to include variables in the model:
for val in x_variables:
    print(f"app_chef_train['{val}'] +")


# In[6]:


# Statsmodel: 

# Merging x and y train:
app_chef_train = pd.concat([X_train, y_train], axis = 1)


# Building the model with selected explanatory variables:
lm_best = smf.ols(formula =  """revenue ~
                                app_chef_train['total_meals'] +
                                app_chef_train['unique_meals'] +
                                app_chef_train['contact_cust_serv'] +
                                app_chef_train['avg_time_vis'] +
                                app_chef_train['late_deliv'] +
                                app_chef_train['avg_prep_time'] +
                                app_chef_train['larger_order'] +
                                app_chef_train['master_classes'] +
                                app_chef_train['total_photos'] +
                                app_chef_train['thr_total_meals'] +
                                app_chef_train['thr_contact_cust_serv'] +
                                app_chef_train['thr_avg_time_vis'] +
                                app_chef_train['thr_late_cancel'] +
                                app_chef_train['thr_total_photos'] +
                                app_chef_train['Negative'] +
                                app_chef_train['Neutral'] +
                                app_chef_train['Positive'] +
                                app_chef_train['business'] +
                                app_chef_train['personal'] +
                                app_chef_train['avg_tckt_order'] +
                                app_chef_train['avg_contact_cust_serv'] +
                                app_chef_train['pct_late_deliv'] +
                                app_chef_train['thr_avg_tckt_order'] +
                                app_chef_train['thr_pct_late_deliv'] +
                                app_chef_train['Satisfied'] +
                                app_chef_train['Unhappy']""", 
                  data = app_chef_train)


# Fitting model:
results = lm_best.fit()

# Summarizing model:
print(results.summary())


# In[7]:


# Scikit-learn models: 
# Instantiating a model object considering all models:
lr = LinearRegression()
ridge_model = sklearn.linear_model.Ridge()
lasso_model = sklearn.linear_model.Lasso( )
#ard_model = sklearn.linear_model.ARDRegression()

# Fitting to the training data per model:
lr_fit = lr.fit(X_train, y_train)
ridge_fit  = ridge_model.fit(X_train, y_train)
lasso_fit = lasso_model.fit(X_train,y_train)
#ard_fit = ard_model.fit(X_train,y_train)

# Predicting new data with test dataset:
lr_pred = lr_fit.predict(X_test)
ridge_pred = ridge_fit.predict(X_test)
lasso_pred = lasso_fit.predict(X_test)
#ard_pred = ard_fit.predict(X_test)

# Generating train and test score per model:
lr_train_score = lr.score(X_train, y_train).round(4)
lr_test_score  = lr.score(X_test, y_test).round(4)
ridge_train_score = ridge_model.score(X_train, y_train).round(4)
ridge_test_score  = ridge_model.score(X_test, y_test).round(4)
lasso_train_score = lasso_model.score(X_train,y_train).round(4)
lasso_test_score  = lasso_model.score(X_test,y_test).round(4)
#ard_train_score = ard_model.score(X_train,y_train).round(4)
#ard_test_score  = ard_model.score(X_test,y_test).round(4)

# Comparing results per model:

print(f"""
Model        Train Score        Test Score
-----        -----------        ----------
OLS          {lr_train_score}              {lr_test_score}
Ridge        {ridge_train_score}             {ridge_test_score}
Lasso        {lasso_train_score}             {lasso_test_score}""" # move it down if including ard model 
#Bayesian ARD {ard_train_score}             {ard_test_score}
)

print("""
Bayesian ARD model was removed from the analysis due the time it takes to run.
Although, it is still on the code. By just removing the '#'s, it is possible 
to include it again.""")


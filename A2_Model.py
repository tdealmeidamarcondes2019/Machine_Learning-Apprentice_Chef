#!/usr/bin/env python
# coding: utf-8

# ***
# ***
# ***
# 
# # Classification Predictive Model<br>
# 
# ***
# ***

# ***
# <h3> Feature Engineering </h3>

# In[ ]:


# Importing necessary packages: 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import random as rand
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# In[ ]:


# Reading external file: 
file = 'Apprentice_Chef_Dataset.xlsx'
app_chef = pd.read_excel(file)


# In[ ]:


# Changing column names: 
# Changing variable names
app_chef = app_chef.rename(columns={'REVENUE'                     :'revenue',
                                    'CROSS_SELL_SUCCESS'          :'cross_sell',
                                    'NAME'                        :'name',
                                    'EMAIL'                       :'email',
                                    'FIRST_NAME'                  :'first_name',
                                    'FAMILY_NAME'                 :'family_name',
                                    'TOTAL_MEALS_ORDERED'         :'total_meals',
                                    'UNIQUE_MEALS_PURCH'          :'unique_meals',
                                    'CONTACTS_W_CUSTOMER_SERVICE' :'contact_cust_serv',
                                    'PRODUCT_CATEGORIES_VIEWED'   :'prod_cat_view',
                                    'AVG_TIME_PER_SITE_VISIT'     :'avg_time_vis',
                                    'MOBILE_NUMBER'               :'mob_number',
                                    'CANCELLATIONS_BEFORE_NOON'   :'early_cancel',
                                    'CANCELLATIONS_AFTER_NOON'    :'late_cancel',
                                    'TASTES_AND_PREFERENCES'      :'tastes_pref',
                                    'PC_LOGINS'                   :'pc_logins',
                                    'MOBILE_LOGINS'               :'mob_logins',
                                    'WEEKLY_PLAN'                 :'week_plan',
                                    'EARLY_DELIVERIES'            :'early_deliv',
                                    'LATE_DELIVERIES'             :'late_deliv',
                                    'PACKAGE_LOCKER'              :'pack_lock',
                                    'REFRIGERATED_LOCKER'         :'ref_lock',
                                    'FOLLOWED_RECOMMENDATIONS_PCT':'follow_rec_pct',
                                    'AVG_PREP_VID_TIME'           :'avg_prep_time',
                                    'LARGEST_ORDER_SIZE'          :'larger_order',
                                    'MASTER_CLASSES_ATTENDED'     :'master_classes',
                                    'MEDIAN_MEAL_RATING'          :'med_meal_rate',
                                    'AVG_CLICKS_PER_VISIT'        :'avg_clicks',
                                    'TOTAL_PHOTOS_VIEWED'         :'total_photos'})


# In[ ]:


# Filling missing values: 
# Creating loop:
for c in app_chef:
    if app_chef[c].isnull().astype(int).sum() > 0:
        app_chef['m_'+c] = app_chef[c].isnull().astype(int)

# Imputing value
app_chef['family_name'] = app_chef['family_name'].fillna('Unknown')


# <strong> New variables </strong><br>

# In[ ]:


# Creating new variables (continuous): 

# Average ticket per order:
app_chef['avg_tckt_order'] = app_chef['revenue']/app_chef['total_meals']
app_chef['avg_tckt_order'] = app_chef['avg_tckt_order'].round(2)

# Average contacts per order:
app_chef['avg_contact_cust_serv'] = app_chef['contact_cust_serv']/app_chef['total_meals']
app_chef['avg_contact_cust_serv'] = app_chef['avg_contact_cust_serv'].round(2)

# Ratio of orders delivered late:
app_chef['pct_late_deliv'] = app_chef['late_deliv']/app_chef['total_meals']
app_chef['pct_late_deliv'] = app_chef['pct_late_deliv'].round(2)

# Ratio of orders delivered early:
app_chef['pct_early_deliv'] = app_chef['early_deliv']/app_chef['total_meals']
app_chef['pct_early_deliv'] = app_chef['pct_early_deliv'].round(2)

# Ratio of unique meals:
app_chef['pct_unique_meals'] = app_chef['unique_meals']/app_chef['total_meals']
app_chef['pct_unique_meals'] = app_chef['pct_unique_meals'].round(2)

# Total logins:
app_chef['total_logins'] = app_chef['mob_logins'] + app_chef['pc_logins']

# Ratio of mobile login:
app_chef['pct_mob_logins'] = app_chef['mob_logins']/app_chef['total_logins']
app_chef['pct_mob_logins'] = app_chef['pct_mob_logins'].round(2)

# Share of total revenue:
app_chef['share_revenue'] = app_chef['revenue']/app_chef['revenue'].sum()
app_chef['share_revenue'] = app_chef['share_revenue'].round(4)

# Share of total meals:
app_chef['share_total_meals'] = app_chef['total_meals']/app_chef['total_meals'].sum()
app_chef['share_total_meals'] = app_chef['share_total_meals'].round(4)


# In[ ]:


# Creating new variables (categorical): 

## 1st: Email domain group

# Creating empty list to fill it later
email_lst = []

# Creating loop to split email address:
for index, col in app_chef.iterrows():
    split_email = app_chef.loc[index, 'email'].split(sep = '@')
    email_lst.append(split_email)

# Filling list and converting into a dataframe:
email_df = pd.DataFrame(email_lst)

# Changing column names:
email_df.columns = ['name','email']

# Looping again to get only the name of the domain (removing '.com'):
email_lst_2 = []
for index, col in email_df.iterrows():
    split_email = email_df.loc[index, 'email'].split(sep = '.')
    email_lst_2.append(split_email)

# Filling list, converting into a DataFrame and changing column names:
email_df_2 = pd.DataFrame(email_lst_2)
email_df_2.columns = ['domain','.com']

# Including column in our main dataset:
app_chef = pd.concat([app_chef, email_df_2['domain']],
                   axis = 1)

# Grouping email domain types:
personal_email_domains = ['gmail', 'protonmail','yahoo']
business_email_domains  = ['amex','merck','mcdonalds','jnj','cocacola','apple',
                          'nike','ibm','ge','dupont','chevron','microsoft','travelers',
                          'unitedhealth','exxon','boeing','caterpillar','mmm','verizon','pg',
                          'disney','walmart','visa','pfizer','jpmorgan','cisco','goldmansacs',
                          'unitedtech','homedepot','intel']
junk_email_domains  = ['me','aol','hotmail','live','msn','passport']

# Looping one more time to massively classify the domains:
email_group = []
for group in app_chef['domain']:
        if group in personal_email_domains:
            email_group.append('personal')
            
        elif group in business_email_domains:
            email_group.append('business')
            
        elif group in junk_email_domains:
            email_group.append('junk')
            
        else:
            print('Unknown')

# Concatenating with our original DataFrame:
app_chef['domain_group'] = pd.Series(email_group)

## 2nd Rating category
# Creating loop to define categories according to ratings
rating_lst = []

for row,col in app_chef.iterrows():
    if app_chef.loc[row,'med_meal_rate'] <= 2:
        rating_lst.append('Negative')
    elif app_chef.loc[row,'med_meal_rate'] >= 4:
        rating_lst.append('Positive')
    else:
        rating_lst.append('Neutral')

# Adding new variable to dataset
app_chef['rating_category'] = pd.Series(rating_lst)


## 3rd Followed recommendations category
# Creating loop to define categories according to ratings
follow_lst = []
for row,col in app_chef.iterrows():
    if app_chef.loc[row,'follow_rec_pct'] <= 30:
        follow_lst.append('Rarely')
    elif app_chef.loc[row,'follow_rec_pct'] <= 70:
        follow_lst.append('Sometimes')
    else:
        follow_lst.append('Frequently')

# Adding new variable to dataset
app_chef['follow_rec_category'] = pd.Series(follow_lst)

## 4th Critical customer (too many contacts on customer service):
app_chef['cust_serv_status'] = 0
good_cust_serv = app_chef.loc[0:,'cust_serv_status'][app_chef['avg_contact_cust_serv'] <= 0.5]
app_chef['cust_serv_status'].replace(to_replace = good_cust_serv,
                                value      = 'Satisfied',
                                inplace    = True)

bad_cust_serv = app_chef.loc[0:,'cust_serv_status'][app_chef['avg_contact_cust_serv'] > 0.5]

app_chef['cust_serv_status'].replace(to_replace = bad_cust_serv,
                                value      = 'Unhappy',
                                inplace    = True)


# In[ ]:


# Creating binary variable: 
# Negative ratings:
app_chef['attended_master_class'] = 0
rate_1 = app_chef.loc[0:,'attended_master_class'][app_chef['master_classes'] > 0]

app_chef['attended_master_class'].replace(to_replace = rate_1,
                                value      = 1,
                                inplace    = True)
rate_2 = app_chef.loc[0:,'attended_master_class'][app_chef['master_classes'] == 0]

app_chef['attended_master_class'].replace(to_replace = rate_2,
                                value      = 0,
                                inplace    = True)


# ***
# <h3> Exploratory analysis </h3>

# In[ ]:


# Setting outlier thresholds: 

# Explanatory variables:
avg_time_hi              = 250
avg_prep_lo              = 70
avg_prep_hi              = 280
total_meals_hi           = 250
unique_meals_hi          = 10
cont_cust_serv_lo        = 2.5
cont_cust_serv_hi        = 12.5
canc_bef_noon_hi         = 6
late_deliv_hi            = 10
larg_order_lo            = 2
larg_order_hi            = 8
avg_clicks_lo            = 8
avg_clicks_hi            = 18
total_photos_hi          = 500
avg_tckt_order_hi        = 80
avg_contact_cust_serv_hi = 0.3
pct_late_deliv_hi        = 0.3
pct_early_deliv_hi       = 0.15
share_total_meals_hi     = 0.0018
pct_unique_meals_hi      = 0.3

# Average visit time:
app_chef['out_avg_time_hi'] = 0
condition_hi = app_chef.loc[0:,'out_avg_time_hi'][app_chef['avg_time_vis'] > avg_time_hi]

app_chef['out_avg_time_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Average preparation time:
## Low:
app_chef['out_avg_prep_lo'] = 0
condition_hi = app_chef.loc[0:,'out_avg_prep_lo'][app_chef['avg_prep_time'] < avg_prep_lo]

app_chef['out_avg_prep_lo'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

## High
app_chef['out_avg_prep_hi'] = 0
condition_hi = app_chef.loc[0:,'out_avg_prep_hi'][app_chef['avg_prep_time'] > avg_prep_hi]

app_chef['out_avg_prep_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Total Meals:
app_chef['out_total_meals_hi'] = 0
condition_hi = app_chef.loc[0:,'out_total_meals_hi'][app_chef['total_meals'] > total_meals_hi]

app_chef['out_total_meals_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Unique Meals:
app_chef['out_unique_meals_hi'] = 0
condition_hi = app_chef.loc[0:,'out_unique_meals_hi'][app_chef['unique_meals'] > unique_meals_hi]

app_chef['out_unique_meals_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Contact customer service:
## High
app_chef['out_cont_cust_serv_lo'] = 0
condition_hi = app_chef.loc[0:,'out_cont_cust_serv_lo'][app_chef['contact_cust_serv'] < cont_cust_serv_lo]

app_chef['out_cont_cust_serv_lo'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

## Low
app_chef['out_cont_cust_serv_hi'] = 0
condition_hi = app_chef.loc[0:,'out_cont_cust_serv_hi'][app_chef['contact_cust_serv'] > cont_cust_serv_hi]

app_chef['out_cont_cust_serv_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Cancelation before noon:
app_chef['out_canc_bef_noon_hi'] = 0
condition_hi = app_chef.loc[0:,'out_canc_bef_noon_hi'][app_chef['late_cancel'] > canc_bef_noon_hi]

app_chef['out_canc_bef_noon_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Late Deliveries:
app_chef['out_late_deliv_hi'] = 0
condition_hi = app_chef.loc[0:,'out_late_deliv_hi'][app_chef['late_deliv'] > late_deliv_hi]

app_chef['out_late_deliv_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Largest order per customer:
## Low
app_chef['out_larg_order_lo'] = 0
condition_hi = app_chef.loc[0:,'out_larg_order_lo'][app_chef['larger_order'] < larg_order_lo]

app_chef['out_larg_order_lo'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

## High
app_chef['out_larg_order_hi'] = 0
condition_hi = app_chef.loc[0:,'out_larg_order_hi'][app_chef['larger_order'] > larg_order_hi]

app_chef['out_larg_order_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Average clicks:
## Low
app_chef['out_avg_clicks_lo'] = 0
condition_hi = app_chef.loc[0:,'out_avg_clicks_lo'][app_chef['avg_clicks'] < avg_clicks_lo]

app_chef['out_avg_clicks_lo'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)
## High
app_chef['out_avg_clicks_hi'] = 0
condition_hi = app_chef.loc[0:,'out_avg_clicks_hi'][app_chef['avg_clicks'] > avg_clicks_hi]

app_chef['out_avg_clicks_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Total Photos:
app_chef['out_total_photos_hi'] = 0
condition_hi = app_chef.loc[0:,'out_total_photos_hi'][app_chef['total_photos'] > total_photos_hi]

app_chef['out_total_photos_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Late Deliveries (% of total):
app_chef['out_pct_late_deliv_hi'] = 0
condition_hi = app_chef.loc[0:,'out_pct_late_deliv_hi'][app_chef['pct_late_deliv'] > pct_late_deliv_hi]

app_chef['out_pct_late_deliv_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Early Deliveries (% of total):
app_chef['out_pct_early_deliv_hi'] = 0
condition_hi = app_chef.loc[0:,'out_pct_early_deliv_hi'][app_chef['pct_early_deliv'] > pct_early_deliv_hi]

app_chef['out_pct_early_deliv_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Total meals:
app_chef['out_share_total_meals_hi'] = 0
condition_hi = app_chef.loc[0:,'out_share_total_meals_hi'][app_chef['share_total_meals'] > share_total_meals_hi]

app_chef['out_share_total_meals_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Response variable:
revenue_hi       = 5000
share_revenue_hi = 0.0013

# Revenue:
app_chef['out_revenue_hi'] = 0
condition_hi = app_chef.loc[0:,'out_revenue_hi'][app_chef['revenue'] > revenue_hi]

app_chef['out_revenue_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Share revenue:
app_chef['out_share_revenue_hi'] = 0
condition_hi = app_chef.loc[0:,'out_share_revenue_hi'][app_chef['share_revenue'] > share_revenue_hi]

app_chef['out_share_revenue_hi'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Ratio unique meals:
app_chef['out_pct_unique_meals'] = 0
condition_hi = app_chef.loc[0:,'out_pct_unique_meals'][app_chef['pct_unique_meals'] > pct_unique_meals_hi]

app_chef['out_pct_unique_meals'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


# In[ ]:


# Creating one binary variable per categorical type: 

# For rating category, we will create a different column per different rank (1 to 5)
# For email domain, we will create one column for 'professional' and another for 'personal'

# Generating binary categorical variables
one_hot_med_meal_rate = pd.get_dummies(app_chef['rating_category'])
one_hot_domain_group = pd.get_dummies(app_chef['domain_group'])
one_hot_cust_serv_status = pd.get_dummies(app_chef['cust_serv_status'])
one_hot_follow_rec_category = pd.get_dummies(app_chef['follow_rec_category'])

# Including binary variables in the dataframe
app_chef = app_chef.join([one_hot_med_meal_rate, one_hot_domain_group, 
                          one_hot_cust_serv_status, one_hot_follow_rec_category])


# ***
# <h3> Classification Model </h3>

# In[ ]:


# Setting explanatory and response variable 

# Explanatory variable:
app_chef_data   = app_chef.drop(['cross_sell'], axis = 1)

# Dropping discrete variables: 
app_chef_data = app_chef_data.drop(['name','email','first_name','family_name',
                                                  'domain', 'domain_group', 'rating_category',
                                                 'cust_serv_status', 'follow_rec_category'],axis = 1)

# Preparing the target variable
app_chef_target = app_chef.loc[ : , 'cross_sell']


# In[ ]:


# Splitting dataset in train and test with stratification:  
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size = 0.25,
            random_state = 222,
            stratify = app_chef_target)

# Creating training dataset for statsmodel:
app_chef_train = pd.concat([X_train, y_train], axis = 1)

# Printing explanatory variables in the format to be included in the model:
#for val in app_chef_data:
#    print(f"{val} +")


# In[ ]:


# Creating empty dataframe to compare models: 
model_performance = []


# <h4> Logistic Regression Model </h4>

# In[ ]:


## Setting Logistic Regression will all explanatory variables: 
# Building model
logistic_full = smf.logit(formula = """ cross_sell ~
                                        revenue +
                                        total_meals +
                                        unique_meals +
                                        contact_cust_serv +
                                        prod_cat_view +
                                        avg_time_vis +
                                        early_cancel +
                                        late_cancel +
                                        tastes_pref +
                                        week_plan +
                                        early_deliv +
                                        late_deliv +
                                        pack_lock +
                                        ref_lock +
                                        follow_rec_pct +
                                        avg_prep_time +
                                        larger_order +
                                        med_meal_rate +
                                        avg_clicks +
                                        total_photos +
                                        m_family_name +
                                        avg_tckt_order +
                                        avg_contact_cust_serv +
                                        pct_late_deliv +
                                        pct_early_deliv +
                                        pct_unique_meals +
                                        total_logins +
                                        pct_mob_logins +
                                        attended_master_class +
                                        out_avg_time_hi +
                                        out_avg_prep_lo +
                                        out_avg_prep_hi +
                                        out_total_meals_hi +
                                        out_unique_meals_hi +
                                        out_late_deliv_hi +
                                        out_larg_order_lo +
                                        out_larg_order_hi +
                                        out_avg_clicks_lo +
                                        out_avg_clicks_hi +
                                        out_total_photos_hi +
                                        out_pct_late_deliv_hi +
                                        out_pct_early_deliv_hi +
                                        out_revenue_hi +
                                        out_pct_unique_meals +
                                        Negative +
                                        Positive +
                                        business +
                                        Unhappy +
                                        Frequently +
                                        Rarely """,
                                        data = app_chef_train)

# We removed the following variables to allow the model to run: 'pc_logins', 'mobile_logins','mob_number','share_revenue',
# 'share_total_meals', 'Neutral','personal','Satisfied','Sometimes'

# Fitting the model object
results_full = logistic_full.fit()


# In[ ]:


## Defining significant variables: 
# Building model only with significant variables (p-value lower than 0.1)
logit_sig = smf.logit(formula = """ cross_sell ~
                                        unique_meals +
                                        early_cancel +
                                        tastes_pref +
                                        follow_rec_pct +
                                        pct_unique_meals +
                                        out_revenue_hi +
                                        out_pct_unique_meals +
                                        pct_mob_logins +
                                        business """,
                                            data    = app_chef_train)

# Fitting:
logit_sig = logit_sig.fit()


# In[ ]:


## Creating a dictionary to store different variable packs: 

candidate_dict = {

 # Model with all explanatory variables:
 'logit_full'   : ['revenue','total_meals','unique_meals','contact_cust_serv','prod_cat_view',
                     'avg_time_vis','early_cancel','late_cancel','tastes_pref','week_plan',
                     'early_deliv','late_deliv','pack_lock','ref_lock','follow_rec_pct','avg_prep_time',
                     'larger_order','med_meal_rate','avg_clicks','total_photos','m_family_name','avg_tckt_order',
                     'avg_contact_cust_serv','pct_late_deliv','pct_early_deliv','pct_unique_meals','total_logins',
                     'pct_mob_logins','attended_master_class','out_avg_time_hi','out_avg_prep_lo','out_avg_prep_hi',
                     'out_total_meals_hi','out_unique_meals_hi','out_late_deliv_hi','out_larg_order_lo',
                     'out_larg_order_hi','out_avg_clicks_lo','out_avg_clicks_hi','out_total_photos_hi',
                     'out_pct_late_deliv_hi','out_pct_early_deliv_hi','out_revenue_hi','out_pct_unique_meals',
                     'Negative','Positive','business','Unhappy','Frequently','Rarely'],
 
 # Model only with significant variables:
 'logit_sig'    : ['unique_meals','early_cancel','tastes_pref','follow_rec_pct','pct_unique_meals',
                     'out_revenue_hi','out_pct_unique_meals','pct_mob_logins','business'],
    
 'tree_sig'     : []

}


# In[ ]:


## Model with all variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_full']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']


# Splitting dataset in train and test:
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Instantiating:
logreg = LogisticRegression(solver = 'lbfgs',
                            C = 1,
                            random_state = 222)


# Fitting:
logreg_fit = logreg.fit(X_train, y_train)


# Predicting:
logreg_pred = logreg_fit.predict(X_test)

# Getting Area Under the ROC Curve (AUC): 
roc_auc_score(y_true  = y_test,
              y_score = logreg_pred)

# Adding model results to consolidated table:
model_performance.append(['Logistic Regression: all',
                          logreg_fit.score(X_train, y_train).round(4),
                          logreg_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                        y_score = logreg_pred).round(4)])


# In[ ]:


## Standardized model with all variables: 
# Divide a standardized data set into train and test variable to run models side by side
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_full']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']

# Instantiating StandardScaler()
scaler = StandardScaler()

# Fitting:
scaler.fit(app_chef_data)


# Tranforming the independent variable data:
X_scaled     = scaler.transform(app_chef_data)


# Converting to a DataFrame:
X_scaled_df  = pd.DataFrame(X_scaled) 


# Spliting dataset again, now with scaled data:
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                    X_scaled_df,
                    app_chef_target,
                    random_state = 222,
                    test_size = 0.25,
                    stratify = app_chef['follow_rec_pct'])
# Instantiating:
logreg = LogisticRegression(solver = 'liblinear',
                            C = 1,
                            random_state = 222)


# Fitting:
logreg_fit = logreg.fit(X_train_scaled, y_train_scaled)


# Predicting:
logreg_pred = logreg_fit.predict(X_test_scaled)

# Adding model results to consolidated table:
model_performance.append(['Logistic Regression: standard/all',
                          logreg_fit.score(X_train_scaled, y_train_scaled).round(4),
                          logreg_fit.score(X_test_scaled, y_test_scaled).round(4),
                          roc_auc_score(y_true  = y_test_scaled,
                                        y_score = logreg_pred).round(4)])


# In[ ]:


## Model with significant variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_sig']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']


# Splitting dataset:
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Instantiating:
logreg = LogisticRegression(solver = 'lbfgs',
                            C = 1,
                            random_state = 222)

# Fitting:
logreg_fit = logreg.fit(X_train, y_train)

# Predicting:
logreg_pred = logreg_fit.predict(X_test)

# Getting Area Under the ROC Curve (AUC): 
roc_auc_score(y_true  = y_test,
              y_score = logreg_pred)

# Adding model results to consolidated table:
model_performance.append(['Logistic Regression: significant',
                          logreg_fit.score(X_train, y_train).round(4),
                          logreg_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                        y_score = logreg_pred).round(4)])


# In[ ]:


## Standardized model with significant variables: 
# Divide a standardized data set into train and test variable to run models side by side
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_sig']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']

# Instantiating:
scaler = StandardScaler()

# Fitting:
scaler.fit(app_chef_data)


# Transforming the independent variable data:
X_scaled     = scaler.transform(app_chef_data)


# Converting to a DataFrame:
X_scaled_df  = pd.DataFrame(X_scaled) 


# Spliting dataset again, now with scaled data:
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                    X_scaled_df,
                    app_chef_target,
                    random_state = 222,
                    test_size = 0.25,
                    stratify = app_chef['follow_rec_pct'])
# Instantiating:
logreg = LogisticRegression(solver = 'liblinear',
                            C = 1,
                            random_state = 222)


# Fitting:
logreg_fit = logreg.fit(X_train_scaled, y_train_scaled)


# Predicting:
logreg_pred = logreg_fit.predict(X_test_scaled)

# Adding model results to consolidated table:
model_performance.append(['Logistic Regression: standard/significant',
                          logreg_fit.score(X_train_scaled, y_train_scaled).round(4),
                          logreg_fit.score(X_test_scaled, y_test_scaled).round(4),
                          roc_auc_score(y_true  = y_test_scaled,
                                        y_score = logreg_pred).round(4)])


# <h4> KNN: Neighbors Regressor Classifier </h4>

# In[ ]:


# Creating function to get optimal_neighbors: 
def optimal_neighbors(X_data,
                      y_data,
                      standardize = True,
                      pct_test=0.25,
                      seed=222,
                      response_type='reg',
                      max_neighbors=20,
                      show_viz=True):    
    
    if standardize == True:
        scaler             = StandardScaler()
        scaler.fit(X_data)
        X_scaled           = scaler.transform(X_data)
        X_scaled_df        = pd.DataFrame(X_scaled)
        X_data             = X_scaled_df

    # Splitting:
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size = pct_test,
                                                        random_state = seed)

    # creating lists for training set accuracy and test set accuracy
    training_accuracy = []
    test_accuracy = []
    
    # Setting neighbor range
    neighbors_settings = range(1, max_neighbors + 1)

    for n_neighbors in neighbors_settings:
        if response_type == 'reg':
            clf = KNeighborsRegressor(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)
            
        elif response_type == 'class':
            clf = KNeighborsClassifier(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)            
            
        else:
            print("Error: response_type must be 'reg' or 'class'")
             
        # Scoring:
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))


    # Plotting accuracy:
    if show_viz == True:
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show() 
    
    # Getting optimal number of neighbors:
    print(f"The optimal number of neighbors is: {test_accuracy.index(max(test_accuracy))+1}")
    return test_accuracy.index(max(test_accuracy))+1


# In[ ]:


## Model with all variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_full']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']


# Note that we used the most significant variable to stratify
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Instantiating:
knn_opt = KNeighborsClassifier(n_neighbors = optimal_neighbors(X_train, 
                                                               y_train,show_viz=False))

# Fitting:
knn_fit = knn_opt.fit(X_train, y_train)

# Predicting:
knn_pred = knn_fit.predict(X_test)

# Adding model results to consolidated table:
model_performance.append(['KNN Classification: all',
                          knn_fit.score(X_train, y_train).round(4),
                          knn_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                          y_score = knn_pred).round(4)])


# In[ ]:


## Standardized model with all variables: 
# Divide a standardized data set into train and test variable to run models side by side
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_full']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']

# Instantiating StandardScaler()
scaler = StandardScaler()

# Fitting:
scaler.fit(app_chef_data)


# Transforming:
X_scaled     = scaler.transform(app_chef_data)


# Converting to a DataFrame:
X_scaled_df  = pd.DataFrame(X_scaled) 


# Spliting dataset again, now with scaled data:
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                    X_scaled_df,
                    app_chef_target,
                    random_state = 222,
                    test_size = 0.25,
                    stratify = app_chef['follow_rec_pct'])

# Instantiating scaled data:
knn_opt = KNeighborsClassifier(n_neighbors = optimal_neighbors(X_train, 
                                                               y_train,
                                                              show_viz = False))


# Fitting:
knn_fit = knn_opt.fit(X_train_scaled, y_train_scaled)


# Predicting:
knn_pred = knn_fit.predict(X_test_scaled)

# Adding model results to consolidated table:
model_performance.append(['KNN Classification: standard/all',
                          knn_fit.score(X_train_scaled, y_train_scaled).round(4),
                          knn_fit.score(X_test_scaled, y_test_scaled).round(4),
                          roc_auc_score(y_true  = y_test_scaled,
                                          y_score = knn_pred).round(4)])


# In[ ]:


## Model with significant variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_sig']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']


# Note that we used the most significant variable to stratify
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Instantiating:
knn_opt = KNeighborsClassifier(n_neighbors = optimal_neighbors(X_train, 
                                                               y_train,
                                                              show_viz = False))


# Fitting:
knn_fit = knn_opt.fit(X_train, y_train)


# Predicting:
knn_pred = knn_fit.predict(X_test)

# Adding model results to consolidated table:
model_performance.append(['KNN Classification: significant',
                          knn_fit.score(X_train, y_train).round(4),
                          knn_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                          y_score = knn_pred).round(4)])


# In[ ]:


## Standardized model with significant variables: 
# Divide a standardized data set into train and test variable to run models side by side
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_sig']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']

# Instantiating StandardScaler():
scaler = StandardScaler()

# Fitting the independent variable data:
scaler.fit(app_chef_data)


# Transforming the independent variable data:
X_scaled     = scaler.transform(app_chef_data)


# Converting to a DataFrame:
X_scaled_df  = pd.DataFrame(X_scaled) 


# Spliting dataset again, now with scaled data:
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                    X_scaled_df,
                    app_chef_target,
                    random_state = 222,
                    test_size = 0.25,
                    stratify = app_chef['follow_rec_pct'])

# Instantiating scaled data:
knn_opt = KNeighborsClassifier(n_neighbors = optimal_neighbors(X_train, 
                                                               y_train,
                                                              show_viz = False))


# Fitting:
knn_fit = knn_opt.fit(X_train_scaled, y_train_scaled)


# Predicting:
knn_pred = knn_fit.predict(X_test_scaled)

# Adding model results to consolidated table:
model_performance.append(['KNN Classification: standard/significant',
                          knn_fit.score(X_train_scaled, y_train_scaled).round(4),
                          knn_fit.score(X_test_scaled, y_test_scaled).round(4),
                          roc_auc_score(y_true  = y_test_scaled,
                                          y_score = knn_pred).round(4)])


# <h4> Decision Tree Classifier </h4>

# In[ ]:


## Model with all variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_full']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']

# Note that we used the most significant variable to stratify
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Building decision tree model:
tree_pruned      = DecisionTreeClassifier(max_depth = 4,
                                          min_samples_leaf = 25,
                                          random_state = 802)

# Fitting:
tree_pruned_fit  = tree_pruned.fit(X_train, y_train)


# Predicting:
tree_pred = tree_pruned_fit.predict(X_test)

# Adding model results to consolidated table:
model_performance.append(['Decision Tree: all',
                          tree_pruned_fit.score(X_train, y_train).round(4),
                          tree_pruned_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                          y_score = tree_pred).round(4)])


# In[ ]:


## Model with significant variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_sig']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']


# Note that we used the most significant variable to stratify
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Building decision tree model:
tree_pruned      = DecisionTreeClassifier(max_depth = 4,
                                          min_samples_leaf = 25,
                                          random_state = 802)

# Fitting:
tree_pruned_fit  = tree_pruned.fit(X_train, y_train)


# Predicting:
tree_pred = tree_pruned_fit.predict(X_test)

# Adding model results to consolidated table:
model_performance.append(['Decision Tree: significant',
                          tree_pruned_fit.score(X_train, y_train).round(4),
                          tree_pruned_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                          y_score = tree_pred).round(4)])


# <h4> Random Forest Classifier </h4>

# In[ ]:


## Model with all variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_full']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']


# Note that we used the most significant variable to stratify
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Instantiating:
rndfor = RandomForestClassifier(criterion = 'gini',
                                bootstrap = True, 
                                max_depth = 4, 
                                n_estimators = 10,
                                min_samples_leaf = 25, 
                                random_state = 222)

# Fitting:
rndfor_fit = rndfor.fit(X_train, y_train)

# Predicting:
rndfor_pred = rndfor_fit.predict(X_test)

# Adding model results to consolidated table:
model_performance.append(['Random Forrest: all',
                          rndfor_fit.score(X_train, y_train).round(4),
                          rndfor_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                          y_score = rndfor_pred).round(4)])


# In[ ]:


## Tunned model with all variables: 
#app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_full']]
#app_chef_target =  app_chef.loc[ : , 'cross_sell']

# Declaring a hyperparameter space:
#estimator_space  = pd.np.arange(100, 1100, 250)
#leaf_space       = pd.np.arange(1, 31, 10)
#criterion_space  = ['gini', 'entropy']
#bootstrap_space  = [True, False]
#warm_start_space = [True, False]


# Creating a hyperparameter grid:
#param_grid = {'n_estimators'     : estimator_space,
#              'min_samples_leaf' : leaf_space,
#              'criterion'        : criterion_space,
#              'bootstrap'        : bootstrap_space,
#              'warm_start'       : warm_start_space}


# Instantiating without hyperparameters:
#full_forest_grid = RandomForestClassifier(random_state = 222)


# GridSearchCV object:
#full_forest_cv = GridSearchCV(estimator  = full_forest_grid,
#                              param_grid = param_grid,
#                              cv         = 3,
#                              scoring    = make_scorer(roc_auc_score,
#                                           needs_threshold = False))


# Fitting:
#full_forest_cv.fit(app_chef_data, app_chef_target)

# Instantiating with hyperparameters:
#full_rf_tuned = RandomForestClassifier(bootstrap        = True,
#                                       criterion        = 'gini',
#                                       min_samples_leaf = 11,
#                                       n_estimators     = 850,
#                                       warm_start       = True,
#                                       random_state     = 222)


# Fitting:
#full_rf_tuned_fit = full_rf_tuned.fit(X_train, y_train)


# Predicting:
#full_rf_tuned_pred = full_rf_tuned_fit.predict(X_test)

# Scoring:
#print('Training ACCURACY:', full_rf_tuned_fit.score(X_train, y_train).round(4))
#print('Testing  ACCURACY:', full_rf_tuned_fit.score(X_test, y_test).round(4))
#print('AUC Score        :', roc_auc_score(y_true  = y_test,
#                                          y_score = full_rf_tuned_pred).round(4))

# Adding model results to consolidated table:
model_performance.append(['Tunned Random Forest: all',
                          0.8273,
                          0.7823,
                          0.7703])


# <h4> Gradient Booster Classifier</h4>

# In[ ]:


## Model with significant variables: 
app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_sig']]
app_chef_target =  app_chef.loc[ : , 'cross_sell']


# Note that we used the most significant variable to stratify
X_train, X_test, y_train, y_test = train_test_split(
            app_chef_data,
            app_chef_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = app_chef['follow_rec_pct'])

# Instantiating:
g_boost = GradientBoostingClassifier(loss = 'deviance',
                                     criterion = 'mae',
                                     learning_rate =  0.1,
                                     n_estimators = 100,
                                     max_features = 3,
                                     random_state  = 222)

# Fitting:
g_boost_fit = g_boost.fit(X_train, y_train)

# Predicting:
g_boost_pred = g_boost_fit.predict(X_test)

# Adding model results to consolidated table:
model_performance.append(['GradientBoosting: significant',
                          g_boost_fit.score(X_train, y_train).round(4),
                          g_boost_fit.score(X_test, y_test).round(4),
                          roc_auc_score(y_true  = y_test,
                                          y_score = g_boost_pred).round(4)])


# In[ ]:


## Tunned model with significant variables: 
#app_chef_data   =  app_chef.loc[ : , candidate_dict['logit_sig']]
#app_chef_target =  app_chef.loc[ : , 'cross_sell']

# Declaring a hyperparameter space:
#learn_space     = pd.np.arange(0.1, 1.6, 0.3)
#estimator_space = pd.np.arange(50, 250, 50)
#depth_space     = pd.np.arange(1, 10)


# Creating a hyperparameter grid:
#param_grid = {'learning_rate' : learn_space,
#              'max_depth'     : depth_space,
#              'n_estimators'  : estimator_space}


# Instantiating without hyperparameters:
#full_gbm_grid = GradientBoostingClassifier(random_state = 222)


# GridSearchCV object:
#full_gbm_cv = GridSearchCV(estimator  = full_gbm_grid,
#                           param_grid = param_grid,
#                           cv         = 3,
#                           scoring    = make_scorer(roc_auc_score,
#                                        needs_threshold = False))


# Fitting:
#full_gbm_cv.fit(app_chef_data, app_chef_target)

# Instantiating with hyperparameters:
#gbm_tuned = GradientBoostingClassifier(learning_rate = 0.1,
#                                       max_depth     = 2,
#                                       n_estimators  = 100,
#                                       random_state  = 222)

# Fitting:
#gbm_tuned_fit = gbm_tuned.fit(X_train, y_train)

# Predicting:
#gbm_tuned_pred = gbm_tuned_fit.predict(X_test)

# Scoring:
#print('Training ACCURACY:', gbm_tuned_fit.score(X_train, y_train).round(4))
#print('Testing  ACCURACY:', gbm_tuned_fit.score(X_test, y_test).round(4))
#print('AUC Score        :', roc_auc_score(y_true  = y_test,
#                                          y_score = gbm_tuned_pred).round(4))

# Adding model results to consolidated table:
model_performance.append(['Tunned Gradient Boosting: significant',
                          0.8136,
                          0.7906,
                          0.7905])


# <h4> Comparing models </h4>

# In[ ]:


# List of all built models with respective results: 
model_performance = pd.DataFrame(model_performance)
model_performance = model_performance.rename(columns={0: 'Model', 
                                                      1: 'Training Accuracy',
                                                      2: 'Testing Accuracy', 
                                                      3: 'AUC Value'})
model_performance


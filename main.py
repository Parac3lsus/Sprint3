import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb


import numpy as np
#train_df = pd.read_csv("application_train.csv", nrows= 1500)
#test_df = pd.read_csv("application_test.csv", nrows = 200)
train_df = pd.read_csv("application_train.csv")
test_df = pd.read_csv("application_test.csv")


#==========================Exploratory Data Analysis=================================#
#================================Dataset Basics======================================#
#=====================================1.0============================================#
print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
#=====================================2.0============================================#
print("==============================================================================")
print(f"Train set columns: {train_df.columns.values}")
print("==============================================================================")
#=====================================3.0============================================#
print("=========================First 5 records Transposed===========================")
pd.set_option('display.max_rows', None)
print(train_df.head(5).transpose())
#=====================================4.0============================================#
print(train_df['TARGET'].value_counts())
#=====================================5.0============================================#
print(train_df.dtypes.value_counts())
#=====================================6.0============================================#
print(train_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0))
#=====================================7.0============================================#
nulls_in_columns = train_df.isna().sum() / len(train_df) * 100
print(nulls_in_columns.sort_values(ascending=False).head(20))
#=====================Analizing distribution of variables============================#
#=====================================1.0============================================#
# plt.figure(figsize=(10, 5))
# sns.histplot(test_df['AMT_CREDIT'], bins=10)
# plt.show()
#nr of bins to 10 or 5
#=====================================2.0============================================#
# feat = 'NAME_EDUCATION_TYPE'
# fg = sns.displot(data=train_df, x=feat, stat='percent', height=3.5, aspect=1.25)
#
# for ax in fg.axes.ravel():
#     # add annotations
#     for c in ax.containers:
#         #custom label calculates percent and add an empty string so 0 value bars don't have a number
#         labels = [f'{w:0.1f}%' if (w := v.get_height()) > 0 else '' for v in c]
#
#         ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=90, padding=2)
#
#     ax.margins(y=0.2)
#
# plt.show()
#=====================================3.0============================================#
# plt.figure(figsize=(15, 10))
# sns.histplot(test_df['OCCUPATION_TYPE'])
#=====================================4.0============================================#
# plt.figure(figsize=(10, 5))
# sns.histplot(test_df['NAME_FAMILY_STATUS'])
#=====================================5.0============================================#
# plt.figure(figsize=(10, 5))
# sns.histplot(test_df['NAME_INCOME_TYPE'])
#================================Preprocessing=======================================#
def get_dummies(train, test):
    lbl_encoder = preprocessing.LabelEncoder()
    oh_encoder = OneHotEncoder(dtype=int, drop='first', sparse=False)
    train_dummies = pd.DataFrame()
    test_dummies = pd.DataFrame()
    for col in train:
        if train[col].dtype == 'object':
            if len(list(train[col].unique())) < 3:
                lbl_encoder.fit(train[col])
                train[col] = lbl_encoder.transform(train[col])
                test[col] = lbl_encoder.transform(test[col])
            else:
                train_dummies = oh_encoder.fit_transform(train[[col]])
                train[oh_encoder.categories_[0][1:]] = train_dummies
                train.drop(col, axis=1, inplace=True)
                test_dummies = oh_encoder.transform(test[[col]])
                test[oh_encoder.categories_[0][1:]] = test_dummies
                test.drop(col, axis=1, inplace=True)
    train_col_list = list(train)
    # We make sure both dataframes have the same rows
    for col in train_col_list:
        if not col in list(test):
            test[col] = 0
    # We make sure rows are in the same order
    test = test[train_col_list]

    return train, test

def replace_outlier_IQR(df):
    q1 = df.quantile(0.95)
    return_df = df.where(df < q1, df.median())
    return return_df

def outliers_correction(train, test):
    #From Kaggle Competition Host: "Value 365243 denotes infinity in days, therefore you can consider them NA values"
    train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    train.replace('XNA', np.nan, inplace=True)
    train.replace('XAP', np.nan, inplace=True)
    test.replace('XNA', np.nan, inplace=True)
    test.replace('XAP', np.nan, inplace=True)

    # We convert only very high values found during data analysis
    train['AMT_INCOME_TOTAL'] = replace_outlier_IQR(train['AMT_INCOME_TOTAL'])
    train['AMT_CREDIT'] = replace_outlier_IQR(train['AMT_CREDIT'])
    train['AMT_GOODS_PRICE'] = replace_outlier_IQR(train['AMT_GOODS_PRICE'])

    return train, test

def impute_values(train, test):
    start_time = time.process_time()
    numerical_features = train.select_dtypes(include='number').columns.tolist()

    string_features = train.select_dtypes(exclude='number').columns.tolist()

    str_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    num_imp = SimpleImputer(missing_values=np.nan, strategy='median')

    train[numerical_features] = num_imp.fit_transform(train[numerical_features])
    train[string_features] = str_imp.fit_transform(train[string_features])

    test[numerical_features] = num_imp.transform(test[numerical_features])
    test[string_features] = str_imp.transform(test[string_features])

    # for col in train:
    #     if train[col].dtype != 'object':
    #         train[col] = train[col].fillna(train[col].median())
    #         test[col] = test[col].fillna(train[col].median())
    #     else:
    #         train[col] = train[col].fillna(train[col].mode()[0])
    #         test[col] = test[col].fillna(train[col].mode()[0])
    print("Time (in seconds) took by Random Forest Fit: ", (time.process_time() - start_time)/10 )  # /10

    return train, test

def data_scaling(train, test):
    scaler = StandardScaler()
    train.set_index('SK_ID_CURR', inplace= True)
    test.set_index('SK_ID_CURR', inplace= True)

    for col in train:
        train[col] = scaler.fit_transform(train[[col]])
        test[col] = scaler.transform(test[[col]])

    return train, test
def splitting(train, test):
    xTrain = train.to_numpy()
    yTrain = xTrain[:, -1]
    xTrain = np.delete(xTrain, -1, axis=1)
    xTest = test.to_numpy()
    return xTrain, xTest, yTrain

def pre_processing(train, test):
    #target_column = pd.DataFrame()
    target_column = train['TARGET']
    train.drop(['TARGET'], axis= 1, inplace=True)
    train, test = outliers_correction(train, test)
    train, test = impute_values(train, test)
    train, test = get_dummies(train, test)
    train, test = data_scaling(train, test)
    train['TARGET'] = target_column.to_numpy()

    return splitting(train, test)

print("=========================Starting Preprocessing===========================")
# train_proc, test_proc = pre_processing(train_df, test_df)
x_train, x_test, y_train = pre_processing(train_df, test_df)
print("=========================Preprocessing Finished===========================")

#================================Models Training===================================#
#==============================LogisticRegression==================================#
#=======================================1==========================================#
log_reg = LogisticRegression(max_iter= 1000, C=0.5, random_state=7)
log_reg.fit(x_train, y_train)
#=======================================2==========================================#
predicts = pd.DataFrame()
predicts['SK_ID_CURR'] = test_df['SK_ID_CURR']
predicts['TARGET'] = log_reg.predict_proba(x_test)[:,1]
predicts.to_csv("predictions_prob",index=False)
print("=========================Predictions Saved================================")
print("=================Logistic Regression Score: 0.73251========================")

#=================================RandomForest===================================#
#=======================================1==========================================#
start_time = time.process_time()
rnd_forest_cl = RandomForestClassifier(random_state=7, n_jobs=-2)
rnd_forest_cl.fit(x_train,y_train)
print("Time (in seconds) took by Random Forest Fit: ", time.process_time() - start_time)#/10

#=======================================2==========================================#
predicts['SK_ID_CURR'] = test_df['SK_ID_CURR']
predicts['TARGET'] = rnd_forest_cl.predict_proba(x_test)[:,1]
predicts.to_csv("random_forest_predictions",index=False)
print("=====================Random Forest Score: 0.68348===========================")

#=====================Randomized Search with Cross Validation======================#
#=======================================1==========================================#
print("Cross Validation Start")
hyperparameter_grid = {
 'bootstrap': [False],
 'max_depth': [10, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': sp.stats.randint(5, 10),
 'min_samples_split': sp.stats.randint(7, 11),
 'n_estimators': sp.stats.randint(270, 310)
}
rnd_forest_cl = RandomForestClassifier(random_state=7, n_jobs=-2)
clf = RandomizedSearchCV(rnd_forest_cl, hyperparameter_grid, random_state=7, cv=None, scoring='roc_auc', n_iter=10, verbose=10)
search = clf.fit(x_train, y_train)
print(search.best_params_)

#=======================================2==========================================#
predicts['SK_ID_CURR'] = test_df['SK_ID_CURR']
predicts['TARGET'] = clf.predict_proba(x_test)[:,1]
predicts.to_csv("random_forest_cv",index=False)
#=======================================3==========================================#
print("===================Random Forest CV Score: 0.72392==========================")

#=============================Training LightGBM Model==============================#
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size = 0.33, random_state = 7, stratify=y_train)

param_grid = {
    'boosting_type': ['gbdt', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}
parameters = {'objective': 'binary',
              'metric' : 'auc',
              'is_unbalance' : 'true',
              'boosting' : 'gbdt',
              'num_leaves' : 63,
              'feature_fraction' : 0.5,
              'bagging_fraction' : 0.5,
              'bagging_freq' : 20,
              'learning_rate' : 0.01,
              'verbose' : -1
            }

train_set = lgb.Dataset(data=train_x, label=train_y)
test_set = lgb.Dataset(data=test_x, label=test_y)

lgb_model = lgb.train(parameters, train_set, valid_sets=test_set, num_boost_round=5000, early_stopping_rounds=50)
print(lgb_model.best_score)

#==================================================================================#
predicts['SK_ID_CURR'] = test_df['SK_ID_CURR']
predicts['TARGET'] = lgb_model.predict(x_test)
predicts.to_csv("lightGBM",index=False)
#==================================================================================#
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

train_df = pd.read_csv("application_train.csv")
test_df = pd.read_csv("application_test.csv")

ids = test_df['SK_ID_CURR']
test_df.drop(['SK_ID_CURR'], axis = 1, inplace=True)
train_df.drop(['SK_ID_CURR'], axis = 1, inplace=True)
target_column = train_df['TARGET']
train_df.drop(['TARGET'], axis=1, inplace=True)


numerical_features = train_df.select_dtypes(include='number').columns.tolist()
string_features = train_df.select_dtypes(exclude='number').columns.tolist()

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])


string_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop = 'if_binary', handle_unknown='ignore', sparse=False))
])

full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('string', string_pipeline, string_features)
])

lr_pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', LogisticRegression())
])

lr_pipeline.fit(train_df, target_column)

predicts = pd.DataFrame()
predicts['SK_ID_CURR'] = ids
predicts['TARGET'] = lr_pipeline.predict_proba(test_df)[:,1]
predicts.to_csv("PipelineRegressor",index=False)

print("Logistic Regressor Pipeline Kaggle Score: 0.73016")


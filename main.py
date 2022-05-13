import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import numpy as np
train_df = pd.read_csv("application_train.csv", nrows= 1500)
test_df = pd.read_csv("application_test.csv", nrows = 200)


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
# sns.histplot(test_df['AMT_CREDIT'])
#=====================================2.0============================================#
# feat = 'NAME_EDUCATION_TYPE'
# fg = sns.displot(data=train_df, x=feat, stat='percent', height=3.5, aspect=1.25)
#
# for ax in fg.axes.ravel():
#     # add annotations
#     for c in ax.containers:
#         # custom label calculates percent and add an empty string so 0 value bars don't have a number
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
    test = train[train_col_list]
    return train, test

def replace_outlier_IQR(df):
    q1 = df.quantile(0.95)
    return_df = df.where(df < q1, df.median())
    return return_df

def outliers_correction(train, test):
    #From Kaggle Competition Host: "Value 365243 denotes infinity in days, therefore you can consider them NA values"
    train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    train.replace('XNA', np.nan, inplace = True)
    train.replace('XAP', np.nan, inplace=True)
    test.replace('XNA', np.nan, inplace = True)
    test.replace('XAP', np.nan, inplace=True)

    # We convert only very high values found during data analysis
    train['AMT_INCOME_TOTAL'] = replace_outlier_IQR(train['AMT_INCOME_TOTAL'])
    train['AMT_CREDIT'] = replace_outlier_IQR(train['AMT_CREDIT'])
    train['AMT_GOODS_PRICE'] = replace_outlier_IQR(train['AMT_GOODS_PRICE'])

    return train, test

def impute_values(train, test):
    for col in train:
        if train[col].dtype != 'object':
            train[col] = train[col].fillna(train[col].median())
            test[col] = test[col].fillna(train[col].median())
        else:
            train[col] = train[col].fillna(train[col].mode()[0])
            test[col] = test[col].fillna(train[col].mode()[0])
    return train, test

def data_scaling(train, test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train.set_index('SK_ID_CURR', inplace= True)
    test.set_index('SK_ID_CURR', inplace= True)
    for col in train:
        train[col] = scaler.fit_transform(train[[col]])
        test[col] = scaler.transform(test[[col]])
    return train, test

def pre_processing(train, test):
    #target_column = pd.DataFrame()
    target_column = train['TARGET']
    train.drop(['TARGET'], axis= 1, inplace=True)

    train, test = outliers_correction(train, test)
    train, test = impute_values(train, test)
    train, test = get_dummies(train, test)
    train, test = data_scaling(train, test)
    train['TARGET'] = target_column.to_numpy()
    return train.to_numpy(), test.to_numpy()

print("=========================Starting Preprocessing===========================")
#proc_train_df, proc_test_df = pre_processing(train_df, test_df)
train_proc, test_proc = pre_processing(train_df, test_df)
print("=========================Preprocessing Finished===========================")




#print(train_df['AMT_CREDIT'].describe())

#pd.set_option('display.max_rows', None)
#print(train_df.head(5).transpose())


# nulls_in_columns = train_df.isna().sum() / len(train_df) * 100
# print(nulls_in_columns.sort_values(ascending=False).head(20))
# print(train_df.select_dtypes(include=np.number).columns.values)
# print("==================================================================")
#print()




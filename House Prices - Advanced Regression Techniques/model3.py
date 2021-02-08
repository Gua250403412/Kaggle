# https://www.kaggle.com/darkside92/detailed-examination-for-house-price-top-10/data#%3E-Don't-forget-to-upvote-if-you-like-my-notebook.-:)
# Detailed Examination for House Price (Top%10)

import numpy
import pandas as pd
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))  # 固定保留5位小数

df_train = pd.read_csv('house-prices/train.csv')  # 读取csv数据
df_test = pd.read_csv('house-prices/test.csv')  # 读取csv数据

# region 数据预处理

# 筛选出相关性强的变量
correlation_train = df_train.corr()
corr_dict = correlation_train['SalePrice'].sort_values(ascending=False).to_dict()
important_columns = []
for key, value in corr_dict.items():
    if ((value > 0.1) & (value < 0.8)) | (value <= -0.1):
        important_columns.append(key)

train_test = pd.concat([df_train, df_test], axis=0, sort=False)  # 合并train和test

# 填充缺失值及新增特征变量
train_test.loc[train_test['Fireplaces'] == 0, 'FireplaceQu'] = 'Nothing'
train_test['LotFrontage'] = train_test['LotFrontage'].fillna(train_test.groupby('1stFlrSF')['LotFrontage'].transform('mean'))
train_test['LotFrontage'].interpolate(method='linear', inplace=True)  # 线性插值
train_test['LotFrontage'] = train_test['LotFrontage'].astype(int)
train_test['MasVnrArea'] = train_test['MasVnrArea'].fillna(train_test.groupby('MasVnrType')['MasVnrArea'].transform('mean'))
train_test['MasVnrArea'].interpolate(method='linear', inplace=True)
train_test['MasVnrArea'] = train_test['MasVnrArea'].astype(int)
train_test["Fence"] = train_test["Fence"].fillna("None")
train_test["FireplaceQu"] = train_test["FireplaceQu"].fillna("None")
train_test["Alley"] = train_test["Alley"].fillna("None")
train_test["PoolQC"] = train_test["PoolQC"].fillna("None")
train_test["MiscFeature"] = train_test["MiscFeature"].fillna("None")
train_test.loc[train_test['BsmtFinSF1'] == 0, 'BsmtFinType1'] = 'Unf'
train_test.loc[train_test['BsmtFinSF2'] == 0, 'BsmtQual'] = 'TA'
train_test['YrBltRmd'] = train_test['YearBuilt'] + train_test['YearRemodAdd']
train_test['Total_Square_Feet'] = (
        train_test['BsmtFinSF1'] + train_test['BsmtFinSF2'] + train_test['1stFlrSF'] + train_test['2ndFlrSF'] + train_test['TotalBsmtSF'])
train_test['Total_Bath'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) + train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))
train_test['Total_Porch_Area'] = (
        train_test['OpenPorchSF'] + train_test['3SsnPorch'] + train_test['EnclosedPorch'] + train_test['ScreenPorch'] + train_test['WoodDeckSF'])
train_test['exists_pool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_garage'] = train_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_fireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_bsmt'] = train_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['old_house'] = train_test['YearBuilt'].apply(lambda x: 1 if x < 1990 else 0)

for i in train_test.columns:
    if 'SalePrice' not in i:
        if 'object' in str(train_test[str(i)].dtype):
            train_test[str(i)] = train_test[str(i)].fillna(method='ffill')
# 标签编码
columns = (
    'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
    'YrSold', 'MoSold', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street',
    'Alley', 'CentralAir', 'MSSubClass', 'OverallCond')
for col in columns:
    lbl_enc = LabelEncoder()
    lbl_enc.fit(list(train_test[col].values))
    train_test[col] = lbl_enc.transform(list(train_test[col].values))
numeric_features = train_test.dtypes[train_test.dtypes != "object"].index
skewed_features = train_test[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)

# 偏度校正
high_skewness = skewed_features[abs(skewed_features) > 0.9]
skewed_features = high_skewness.index
for feature in skewed_features:
    train_test[feature] = boxcox1p(train_test[feature], boxcox_normmax(train_test[feature] + 1))

train_test = pd.get_dummies(train_test, dtype='int8')  # 数据编码

train = train_test[0:1460]
test = train_test[1460:2919]

train.interpolate(method='linear', inplace=True)
test.interpolate(method='linear', inplace=True)

# 2次筛选变量
corr_dict2 = train.corr()['SalePrice'].sort_values(ascending=False).to_dict()
best_columns = []
for key, value in corr_dict2.items():
    if ((value >= 0.3175) & (value < 0.9)) | (value <= -0.315):
        best_columns.append(key)
train['SalePrice_Log1p'] = numpy.log1p(train.SalePrice)

# 寻找异常值
rbst_scaler = RobustScaler()
train_rbst = rbst_scaler.fit_transform(train)
train_pca = PCA(3).fit_transform(train_rbst)

dbscan = DBSCAN(eps=1400, min_samples=20).fit(train_pca)
core_samples_mask = numpy.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

labels = pd.DataFrame(labels, columns=['Classes'])
train = pd.concat([train, labels], axis=1)

train.drop([197, 810, 1170, 1182, 1298, 1386, 1423], axis=0, inplace=True)

# 筛选大聚合度的点
train = train[train.GarageArea * train.GarageCars < 3700]
train = train[(train.FullBath + (train.HalfBath * 0.5) + train.BsmtFullBath + (train.BsmtHalfBath * 0.5)) < 5]

del test['SalePrice']
# endregion

X = train.drop(['SalePrice', 'SalePrice_Log1p', 'Classes'], axis=1)
y = train.SalePrice_Log1p

# 去除过拟合的特征变量
overfitted_features = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.9:
        overfitted_features.append(i)
overfitted_features = list(overfitted_features)
X.drop(overfitted_features, axis=1, inplace=True)
test.drop(overfitted_features, axis=1, inplace=True)

# 构建预测模型
std_scaler = StandardScaler()
rbst_scaler = RobustScaler()
power_transformer = PowerTransformer()
X_std = std_scaler.fit_transform(X)
X_rbst = rbst_scaler.fit_transform(X)
X_pwr = power_transformer.fit_transform(X)

test_std = std_scaler.transform(test)
test_rbst = rbst_scaler.transform(test)
test_pwr = power_transformer.transform(test)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.002, random_state=52)

lgb_regressor = LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.035, n_estimators=2177, max_bin=50, bagging_fraction=0.65,
                              bagging_freq=5, bagging_seed=7, feature_fraction=0.201, feature_fraction_seed=7, n_jobs=-1)
lgb_regressor.fit(X_train, y_train)

gb_reg = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03005, max_depth=4, max_features='sqrt', min_samples_leaf=15,
                                   min_samples_split=14, loss='huber', random_state=42)
gb_reg.fit(X_train, y_train)

kfolds = KFold(n_splits=8, shuffle=True, random_state=42)

alphas = [1e-9, 1e-8, 1e-7, 1e-6]
ridgecv_reg = make_pipeline(RidgeCV(alphas=alphas, cv=kfolds))
ridgecv_reg.fit(X_train, y_train)

lassocv_reg = make_pipeline(LassoCV(alphas=alphas, cv=kfolds))
lassocv_reg.fit(X_train, y_train)

alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
l1ratio = [0.87, 0.9, 0.92, 0.95, 0.97, 0.99, 1]
elasticv_reg = make_pipeline(ElasticNetCV(alphas=alphas, cv=kfolds, l1_ratio=l1ratio))
elasticv_reg.fit(X_train, y_train)

estimators = [('lgbm', lgb_regressor), ('gbr', gb_reg), ('lasso', lassocv_reg), ('ridge', ridgecv_reg), ('elasticnet', elasticv_reg)]

stack_reg = StackingRegressor(estimators=estimators, final_estimator=ExtraTreesRegressor(n_estimators=50), n_jobs=-1)
stack_reg.fit(X_train, y_train)

test_pred_lgb = lgb_regressor.predict(test_pwr)
test_pred_ridge = ridgecv_reg.predict(test_pwr)
test_pred_stack = stack_reg.predict(test_pwr)
test_pred_lgb = pd.DataFrame(test_pred_lgb, columns=['SalePrice'])
test_pred_ridge = pd.DataFrame(test_pred_ridge, columns=['SalePrice'])
test_pred_stack = pd.DataFrame(test_pred_stack, columns=['SalePrice'])
test_pred_lgb.SalePrice = numpy.floor(numpy.expm1(test_pred_lgb.SalePrice))
test_pred_ridge.SalePrice = numpy.floor(numpy.expm1(test_pred_ridge.SalePrice))
test_pred_stack.SalePrice = numpy.floor(numpy.expm1(test_pred_stack.SalePrice))

final_pred = (test_pred_stack * 0.1665) + (test_pred_lgb * 0.678) + (test_pred_ridge * 0.1665)

sample_sub = pd.read_csv('house-prices/sample_submission.csv')
sample_sub['SalePrice'] = final_pred
sample_sub.to_csv('SampleSubmissionForHousePrice.csv', index=False)

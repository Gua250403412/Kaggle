import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv('house-prices/train.csv')  # 读取csv数据
df_test = pd.read_csv('house-prices/test.csv')  # 读取csv数据

# region 数据预处理
y = numpy.copy(df_train['SalePrice'].values)
y = y.astype(int)

df_train = df_train[
    ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
     'GarageYrBlt', 'MasVnrArea', 'Fireplaces']]
df_test = df_test[
    ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
     'GarageYrBlt', 'MasVnrArea', 'Fireplaces']]
full_data = [df_train, df_test]
for df in full_data:
    df['GarageCars'] = df['GarageCars'].apply(lambda x: 0 if x != x else x)

    df['GarageArea'] = df['GarageArea'].apply(lambda x: 0 if x != x else x)

    mean = df['TotalBsmtSF'].mean()
    df['TotalBsmtSF'] = df['TotalBsmtSF'].apply(lambda x: mean if x != x else x)

    median = df['TotalBsmtSF'].median()
    df['GarageYrBlt'] = df['GarageYrBlt'].apply(lambda x: median if x != x else x)

    mean = df['MasVnrArea'].mean()
    df['MasVnrArea'] = df['MasVnrArea'].apply(lambda x: mean if x != x else x)

X = numpy.copy(df_train.values)
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# endregion

# clf = GradientBoostingRegressor(n_estimators=100, random_state=0)
clf = RandomForestRegressor(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
print(mean_squared_error(y_test, clf.predict(X_test), squared=False))

# X_final = numpy.copy(df_test.values)
# X_final = X_final.astype(float)
# predict = clf.predict(X_final)
#
# df_result = pd.read_csv('house-prices/sample_submission.csv')  # 读取csv数据
# for i in range(len(df_result)):
#     df_result['SalePrice'][i] = predict[i]
# df_result.to_csv('submission.csv', index=False)

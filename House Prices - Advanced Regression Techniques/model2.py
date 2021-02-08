import numpy
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))  # 固定保留5位小数

df_train = pd.read_csv('house-prices/train.csv')  # 读取csv数据
df_test = pd.read_csv('house-prices/test.csv')  # 读取csv数据
test_ID = df_test['Id']

# region 数据预处理

# 去除异常值
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1183].index)
df_train = df_train.drop(df_train[df_train['Id'] == 692].index)
df_train = df_train.drop(df_train[df_train['Id'] == 298].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1170].index)

# 提取目标值
y_train = numpy.copy(df_train['SalePrice'].values)
y_train = y_train.astype(float)
y_train = numpy.log1p(y_train)  # 使用log1p对数据做正态转换
df_train = df_train.drop(['SalePrice'], axis=1)

# 筛选变量
df_train = df_train[
    ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
     'GarageYrBlt', 'MasVnrArea', 'Fireplaces']]
df_test = df_test[
    ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
     'GarageYrBlt', 'MasVnrArea', 'Fireplaces']]

full_data = [df_train, df_test]
for df in full_data:
    # 填充缺失值
    for col in ('MasVnrArea', 'TotalBsmtSF', 'GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)  # .apply(lambda x: 0 if x != x else x)
# endregion

n_train = df_train.shape[0]
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data = pd.get_dummies(all_data)
df_train = all_data[:n_train]
df_test = all_data[n_train:]
kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(df_train.values)


# clf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
#                                 loss='huber', random_state=0)
# clf = RandomForestRegressor(n_estimators=1000, random_state=0)
# rmse = numpy.sqrt(-cross_val_score(clf, df_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
# print("Score: {:.4f} ({:.4f})".format(rmse.mean(), rmse.std()))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = numpy.column_stack([
            model.predict(X) for model in self.models_
        ])
        return numpy.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = numpy.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = numpy.column_stack([
            numpy.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=0))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0))
rfg = RandomForestRegressor(n_estimators=1000, random_state=0)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=0)
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,
                   reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, random_state=7, nthread=-1)
lgb = LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin=55, bagging_fraction=0.8, bagging_freq=5,
                    feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

# averaged_models = AveragingModels(models = (ENet, GBoost, rfg, lasso))
# rmse = numpy.sqrt(-cross_val_score(averaged_models, df_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
# # print("Score: {:.4f} ({:.4f})".format(rmse.mean(), rmse.std()))

stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
rmse = numpy.sqrt(-cross_val_score(stacked_averaged_models, df_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
print("Score: {:.4f} ({:.4f})".format(rmse.mean(), rmse.std()))

# clf.fit(df_train.values, y_train)
# predict = numpy.expm1(clf.predict(df_test.values))
#
# df_result = pd.read_csv('house-prices/sample_submission.csv')  # 读取csv数据
# for i in range(len(df_result)):
#     df_result['SalePrice'][i] = predict[i]
# df_result.to_csv('submission.csv', index=False)

# stacked_averaged_models.fit(df_train.values, y_train)
# stacked_pred = numpy.expm1(stacked_averaged_models.predict(df_test.values))
#
# xgb.fit(df_train, y_train)
# xgb_pred = numpy.expm1(xgb.predict(df_test))
#
# lgb.fit(df_train, y_train)
# lgb_pred = numpy.expm1(lgb.predict(df_test.values))

# ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
# sub = pd.DataFrame()
# sub['Id'] = test_ID
# sub['SalePrice'] = ensemble
# sub.to_csv('submission.csv', index=False)

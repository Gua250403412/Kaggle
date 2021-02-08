import numpy
import pandas as pd
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))  # 固定保留5位小数

df_train = pd.read_csv('tabular-playground-series-feb-2021/train.csv')  # 读取csv数据
df_test = pd.read_csv('tabular-playground-series-feb-2021/test.csv')  # 读取csv数据
test_ID = df_test['id']
# region 数据预处理
# cat0,cat1,cat2:A/B
# cat3,cat4,cat5:A/B/C/D
# cat6:A-I
# cat7,cat8:A-G
# cat9:A-N
full_data = [df_train, df_test]
for df in full_data:
    df.drop(['id'], axis=1, inplace=True)

    for col in ['cat' + str(i) for i in range(10)]:
        labelEncoder = LabelEncoder()
        labelEncoder.fit(list(df[col].values))
        df[col] = labelEncoder.transform(list(df[col].values))

# endregion

# region 构建模型
X = df_train.drop(['target'], axis=1)
y = df_train['target']


def objective(trial, data=X, target=y):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    param = {
        'metric': 'rmse',
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10, 20, 100]),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100)
    }
    model = LGBMRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)

    return rmse


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)

params = {'reg_alpha': 6.147694913504962,
          'reg_lambda': 0.002457826062076097,
          'colsample_bytree': 0.3,
          'subsample': 0.8,
          'learning_rate': 0.008,
          'max_depth': 20,
          'num_leaves': 111,
          'min_child_samples': 285,
          'random_state': 48,
          'n_estimators': 20000,
          'metric': 'rmse',
          'cat_smooth': 39}

k = KFold(n_splits=5, shuffle=True, random_state=42)

# X_std = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=52)

lgb = LGBMRegressor(**params)
lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
scores = -cross_val_score(lgb, X_test, y_test, cv=k, scoring='neg_root_mean_squared_error')
print(scores, scores.mean())

# xgb = XGBRegressor()
# xgb.fit(X_train, y_train)
# scores = -cross_val_score(xgb, X_test, y_test, cv=5, scoring='neg_root_mean_squared_error')
# print(scores, scores.mean())
# endregion

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['target'] = lgb.predict(df_test.values)
sub.to_csv('submission.csv', index=False)

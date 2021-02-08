import numpy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))  # 固定保留5位小数

df_train = pd.read_csv('tabular-playground-series-feb-2021/train.csv')  # 读取csv数据
df_test = pd.read_csv('tabular-playground-series-feb-2021/test.csv')  # 读取csv数据
id = df_train['id']

# region 相关性热力图
# correlation matrix
pearson_corr = df_train.corr(method='pearson')
spearman_corr = df_train.corr(method='spearman')

# target correlation matrix
# plt.figure(figsize=(16, 9), dpi=300)
# k = 15  # number of variables for heatmap
# cols = pearson_corr.nlargest(k, 'target')['target'].index
# print(cols)
# cm = numpy.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.title('Pearson Correlation of Top ' + str(k) + ' Features', y=1.05, size=15)
# plt.show()

# plt.figure(figsize=(16, 9), dpi=300)
# cols = df_train.corr(method='spearman').nlargest(k, 'target')['target'].index
# print(cols)
# cm = numpy.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.title('Spearman Correlation of Top ' + str(k) + ' Features', y=1.05, size=15)
# plt.show()
# endregion

pass
# # region target分布特性
# # 查看target分布特性
# fig1 = plt.figure()
# (mu, sigma) = norm.fit(df_train['target'])
# sns.displot(df_train['target'], kde=True, height=10, aspect=1.5)
# plt.legend(['正态曲线拟合（$\mu=${:.2f}、$\sigma=${:.2f}）'.format(mu, sigma)], loc='best')
# plt.ylabel('数量')
# plt.title('target柱状图')
# plt.show()
#
# # 绘制QQ图
# fig2 = plt.figure(figsize=(16, 9))
# stats.probplot(df_train['target'], plot=plt)
# plt.show()
#
# # 使用log1p对数据做正态转换
# df_train["target"] = numpy.log1p(df_train["target"])
#
# # 查看target的正态性
# fig3 = plt.figure()
# (mu, sigma) = norm.fit(df_train['target'])
# sns.displot(df_train['target'], kde=True, height=10, aspect=1.5)
# plt.legend(['正态曲线拟合（$\mu=${:.2f}、$\sigma=${:.2f}）'.format(mu, sigma)], loc='best')
# plt.ylabel('数量')
# plt.title('target柱状图')
#
# fig4 = plt.figure(figsize=(16, 9))
# stats.probplot(df_train['target'], plot=plt)
# plt.show()
# # endregion

pass

# all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
# all_data_na = all_data.isnull().sum()
# all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index)

# # region 筛选异常值
# scatterplot
# sns.set()
# cols = ['target', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], height=3, aspect=1.2)
# plt.show()

# df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
# df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# df_train = df_train.drop(df_train[df_train['Id'] == 1183].index)
# df_train = df_train.drop(df_train[df_train['Id'] == 692].index)
# df_train = df_train.drop(df_train[df_train['Id'] == 298].index)
# df_train = df_train.drop(df_train[df_train['Id'] == 1170].index)

# cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
#         'GarageYrBlt', 'MasVnrArea', 'Fireplaces']
# var = 'Fireplaces'
# data = pd.concat([df_train['target'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='target')
# plt.show()
# # endregion

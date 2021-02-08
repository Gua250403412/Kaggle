import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('titanic/train.csv')  # 读取csv数据
df_test = pd.read_csv('titanic/test.csv')  # 读取csv数据

# region 数据预处理
full_data = [df_train, df_test]
for df in full_data:
    # 性别编码
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # 填充年龄缺失值
    age = df['Age'].median()
    df['Age'] = df['Age'].apply(lambda x: age if x != x else x).astype(int)
    # 填充船费=0的值
    fare = df['Fare'].median()
    df['Fare'] = df['Fare'].apply(lambda x: fare if x == 0 or x != x else x)
    # 人名编码
    df['Name_Length'] = df['Name'].apply(lambda x: len(x.split()))  # 根据名字长度增加新的特征
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    df['Title'] = df['Title'].fillna(0)
    # 根据是否有船舱增加新特征
    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    # 根据船票信息增加新特征
    df['Ticket_type'] = df['Ticket'].apply(lambda x: x[:])
    df['Ticket_type'] = df['Ticket_type'].astype('category')
    df['Ticket_type'] = df['Ticket_type'].cat.codes
    # 上船点编码（人工找到众数为S，将nan值替换为S）
    df['Embarked'] = df['Embarked'].apply(lambda x: 'S' if x != x else x)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    # 根据亲属信息增加新特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x > 1 else 0)

df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp', 'FamilySize'], axis=1)
# endregion

data = df_train.values
X = numpy.copy(data[:, 1:])
X = X.astype(float)
y = numpy.copy(data[:, 0])
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

data_final = df_test.values
X_final = numpy.copy(data_final)
X_final = X_final.astype(float)
predict = clf.predict(X_final)

df_result = pd.read_csv('titanic/gender_submission.csv')  # 读取csv数据
for i in range(len(df_result)):
    df_result['Survived'][i] = predict[i]
df_result.to_csv('submission.csv', index=False)

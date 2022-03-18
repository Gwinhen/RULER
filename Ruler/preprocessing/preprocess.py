# This is preprocess of dataset compas
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# 不用独热编码, 而是采用和 EIDIG 相同的策略, 分类变量全部转换为 numerical
# 转换的方式采用 scikit-learn.preprocess.LabelEncoder
def transform_dataset(df):
    """
    :param df:
    :return: Tuple of the transformed dataset and the labels Y and S
    """
    # print("Input is \n{}".format([col for col in df]))

    df_binary = df[(df["race"] == "Caucasian") | (df["race"] == "African-American")]

    del df_binary['c_jail_in']
    del df_binary['c_jail_out']

    # separated class from the rests of the features
    # remove unnecessary dimensions from Y -> only the decile_score remains
    Y = df_binary['decile_score'].to_numpy()
    # we adopt the threshold from ori_data, >=4 , means there's more likely to be a recidivism
    Y = np.apply_along_axis(lambda x: x >= 4, 0, Y).astype(np.int32)

    del df_binary['decile_score']
    del df_binary['two_year_recid']
    del df_binary['score_text']

    # 查看一下每个变量分别是什么类型:
    # print(df_binary.dtypes)
    # 0(sex), 1(age_cat), 2(race), 10(c_charge_degree) 需要转成分类变量

    label_encoder = preprocessing.LabelEncoder()
    data_to_encode = df_binary.to_numpy()
    # print('Before \n {}'.format(data_to_encode[:, 2]))
    # print('numpy is \n{}'.format(data_to_encode))
    for index in [0, 1, 2, 10]:
        data_to_encode[:, index] = label_encoder.fit_transform(data_to_encode[:, index])
    Y = label_encoder.fit_transform(Y)
    # print('After \n {}'.format(data_to_encode[:, 2]))
    # constraint = np.vstack((data_to_encode.min(axis=0), data_to_encode.max(axis=0))).T
    # print(constraint)
    return data_to_encode, Y


# load compas dataset
data_path = '../data/PGD_dataset/original_data/compas.csv'
out_path = '../data/PGD_dataset/compas'
df = pd.read_csv(data_path, encoding='latin-1')  # 6127*17
print('ori data shape is {}'.format(df.shape))

# df_binary 6172*17 <class 'tuple'>;其他都是 Series
X, y = transform_dataset(df)  # FN 就是这样的调用接口

X = X.astype(np.int32)
y = y.astype(np.int32)
print('transformed data shape is {}'.format(X.shape))
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# set constraints for each attribute, 117936000 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

# protected_attributes are race [5]
protected_attribs = [2]
np.save('{}/constraint.npy'.format(out_path), constraint)
np.save('{}/x_train.npy'.format(out_path), X_train)
np.save('{}/y_train.npy'.format(out_path), y_train)
np.save('{}/x_val.npy'.format(out_path), X_val)
np.save('{}/y_val.npy'.format(out_path), y_val)
np.save('{}/x_test.npy'.format(out_path), X_test)
np.save('{}/y_test.npy'.format(out_path), y_test)
np.save('{}/protected_attribs.npy'.format(out_path), protected_attribs)

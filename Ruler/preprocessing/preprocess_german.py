"""
This python file preprocesses the German Credit Dataset.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Q8MAW8
"""

# make outputs stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# load german credit risk dataset
data_path = '../data/PGD_dataset/original_data/german.csv'
out_path = '../data/PGD_dataset/german'
df = pd.read_csv(data_path)

# preprocess data
data = df.values.astype(np.int32)
print([c for c in df])
data[:, 0] = (data[:, 0] == 1).astype(np.int64)
bins_loan_nurnmonth = [0] + [np.percentile(data[:, 2], percent, axis=0) for percent in [25, 50, 75]] + [80]
bins_creditamt = [0] + [np.percentile(data[:, 4], percent, axis=0) for percent in [25, 50, 75]] + [200]
bins_age = [15, 25, 45, 65, 120]
list_index_num = [2, 4, 10]
list_bins = [bins_loan_nurnmonth, bins_creditamt, bins_age]
for index, bins in zip(list_index_num, list_bins):
    data[:, index] = np.digitize(data[:, index], bins, right=True)

# split data into training data, validation data and test data
X = data[:, 1:]
y = data[:, 0]
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# set constraints for each attribute, 839808 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

# for german credit data, gender(6) and age(9) are protected attributes in 24 features
protected_attribs = [6, 9]


np.save('{}/constraint.npy'.format(out_path), constraint)
np.save('{}/x_train.npy'.format(out_path), X_train)
np.save('{}/y_train.npy'.format(out_path), y_train)
np.save('{}/x_val.npy'.format(out_path), X_val)
np.save('{}/y_val.npy'.format(out_path), y_val)
np.save('{}/x_test.npy'.format(out_path), X_test)
np.save('{}/y_test.npy'.format(out_path), y_test)
np.save('{}/protected_attribs.npy'.format(out_path), protected_attribs)

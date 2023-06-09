''' This config file will handle the video anomaly detection with dictionary learning (dl) '''

from ..base import *
from munch import DefaultMunch

dataset = 'ucf-crime'

data_file_train = f'data/{dataset}/{dataset}.training.csv' # video list
data_file_test = f'data/{dataset}/{dataset}.testing.csv' # video list
# ann_file_test = f'data/{dataset}/{dataset}_ground_truth.testing.json' # ground truth

ann_file_test = f'data/{dataset}/qy_{dataset}_ground_truth.testing.json'


random_state = 49
init_lr = 0.001

# dataset settings
base_dict = dict(
    dataset=dataset,
    data_root=data_root,
    backbone=backbone,
    quantize_size=quantize_size,
    data_file=None,
    ann_file=None)

train_regular_dict = base_dict.copy()
train_anomaly_dict = base_dict.copy()
test_dict = base_dict.copy()

train_regular_dict.update(dict(test_mode=False, is_normal=True))
train_anomaly_dict.update(dict(test_mode=False, is_normal=False))
test_dict.update(dict(test_mode=True, is_normal=False))

data = dict(
    train=dict(
        regular=train_regular_dict,
        anomaly=train_anomaly_dict),
    test=test_dict)

data = DefaultMunch.fromDict(data)

data.train.regular.dataset = dataset
data.train.anomaly.dataset = dataset
data.test.dataset = dataset

data.train.regular.data_file = data_file_train
data.train.anomaly.data_file = data_file_train
data.test.data_file = data_file_test
data.test.ann_file = ann_file_test



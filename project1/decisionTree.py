# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2021-11-20 15-03
@file: decisionTree.py
"""
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter


origin_data = np.loadtxt("./kddtrain2021.txt")
sample_len = origin_data.shape[0]

n_fold = 10


def fit_by_decision_tree(n_splits=1, criterion="gini", max_depth=5):
    """
    基于决策树进行拟合
    :param n_splits: N折交叉验证的折数
    :param criterion: 分割属性的评价指标
    :param max_depth: 决策树的最大深度
    :return: N折交叉验证的平均训练准确率、测试准确率
    """
    k_fold = KFold(n_splits=n_splits)
    train_acc_history, test_acc_history = [], []
    for train_index, test_index in k_fold.split(origin_data):
        x_train, y_train = origin_data[train_index, 0:-1], origin_data[train_index, -1]
        x_test, y_test = origin_data[test_index, 0:-1], origin_data[test_index, -1]
        decision_tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        decision_tree.fit(x_train, y_train)
        train_acc = decision_tree.score(x_train, y_train)
        test_acc = decision_tree.score(x_test, y_test)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
    return np.mean(train_acc_history), np.mean(test_acc_history)


writer = SummaryWriter(log_dir=f'DecisionTree', flush_secs=2)
log_value = writer.add_scalars


for criterion in ["gini", "entropy"]:
    for depth in range(5, 20):
        train_acc, test_acc = fit_by_decision_tree(criterion=criterion, n_splits=n_fold, max_depth=depth)
        log_value(f"Accuracy-{criterion}", {"train": train_acc, "test": test_acc}, depth)

# entropy depth=16
# gini depth=10



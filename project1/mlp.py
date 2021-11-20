# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2021-11-20 18-56
@file: mlp.py
"""
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter


origin_data = np.loadtxt("./kddtrain2021.txt")
sample_len = origin_data.shape[0]


def fit_by_mlp(n_splits=1, hidden_layer_sizes=(100,), activation="relu", solver="adam", alpha=0.0001, batch_size="auto"):
    """
    基于MLP进行拟合
    :param n_splits: N折交叉验证的折数
    :param hidden_layer_sizes:
    :param activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    :param solver: {‘lbfgs’, ‘sgd’, ‘adam’}
    :param alpha: L2 penalty (regularization term) parameter.
    :param batch_size: 批量大小
    :return: N折交叉验证的平均训练准确率、测试准确率
    """
    k_fold = KFold(n_splits=n_splits)
    train_acc_history, test_acc_history = [], []
    for train_index, test_index in k_fold.split(origin_data):
        x_train, y_train = origin_data[train_index, 0:-1], origin_data[train_index, -1]
        x_test, y_test = origin_data[test_index, 0:-1], origin_data[test_index, -1]
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                            batch_size=batch_size)
        mlp.fit(x_train, y_train)
        y_hat = mlp.predict(x_test)
        y_train_predict = mlp.predict(x_train)
        test_acc_history.append(accuracy_score(y_test, y_hat))
        train_acc_history.append(accuracy_score(y_train, y_train_predict))
    return np.mean(train_acc_history), np.mean(test_acc_history)


n_fold = 10
writer = SummaryWriter(log_dir=f'MLP', flush_secs=2)
log_value = writer.add_scalars
for activation in ["logistic", "tanh", "relu"]:
    for solver in ["sgd", "adam"]:
        for hidden_layer_size in range(10, 200, 20):
            train_acc, test_acc = fit_by_mlp(n_fold, hidden_layer_sizes=(hidden_layer_size,), activation=activation,
                                             solver=solver)
            log_value(f"Accuracy-{activation}-{solver}", {"train": train_acc, "test": test_acc}, hidden_layer_size)



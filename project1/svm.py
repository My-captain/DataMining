# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2021-11-19 20-42
@file: svm.py
"""
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter


origin_data = np.loadtxt("./kddtrain2021.txt")
sample_len = origin_data.shape[0]


def fit_by_svm(n_splits=1, kernel="rbf", degree=3):
    """
    基于SVM进行拟合
    :param n_splits: N折交叉验证的折数
    :param kernel:
        Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
        ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to
        pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples)
    :param degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    :return: N折交叉验证的平均训练准确率、测试准确率
    """
    k_fold = KFold(n_splits=n_splits)
    train_acc_history, test_acc_history = [], []
    for train_index, test_index in k_fold.split(origin_data):
        x_train, y_train = origin_data[train_index, 0:-1], origin_data[train_index, -1]
        x_test, y_test = origin_data[test_index, 0:-1], origin_data[test_index, -1]
        clf = SVC(decision_function_shape='ovo', kernel=kernel, degree=degree)
        clf.fit(x_train, y_train)
        y_hat = clf.predict(x_test)
        y_train_predict = clf.predict(x_train)
        test_acc_history.append(accuracy_score(y_test, y_hat))
        train_acc_history.append(accuracy_score(y_train, y_train_predict))
    return np.mean(train_acc_history), np.mean(test_acc_history)


n_fold = 10
writer = SummaryWriter(log_dir=f'logs/SVM', flush_secs=2)
log_value = writer.add_scalars
for kernel_func in ["linear", "poly", "rbf", "sigmoid"]:
    for degree in range(6):
        if kernel_func != "poly" and degree > 0:
            break
        train_acc, test_acc = fit_by_svm(n_fold, kernel=kernel_func, degree=degree)
        log_value(f"Accuracy-{kernel_func}", {"train": train_acc, "test": test_acc}, degree)

# rbf 0.9442
# poly degree=4 0.952
# sigmoid 0.3517


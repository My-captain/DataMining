# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2021-11-20 18-30
@file: knn.py
"""
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter


origin_data = np.loadtxt("./kddtrain2021.txt")
sample_len = origin_data.shape[0]


def fit_by_knn(n_splits=1, n_neighbors=5, weights="uniform", algorithm="auto"):
    """
    基于SVM进行拟合
    :param n_splits: N折交叉验证的折数
    :param n_neighbors: Number of neighbors to use by default for kneighbors queries.
    :param weights: Weight function used in prediction. Possible values:
                ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors
    :param algorithm: Algorithm used to compute the nearest neighbors:
                ‘ball_tree’ will use BallTree
                ‘kd_tree’ will use KDTree
                ‘brute’ will use a brute-force search.
                ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method
    :return: N折交叉验证的平均训练准确率、测试准确率
    """
    k_fold = KFold(n_splits=n_splits)
    train_acc_history, test_acc_history = [], []
    for train_index, test_index in k_fold.split(origin_data):
        x_train, y_train = origin_data[train_index, 0:-1], origin_data[train_index, -1]
        x_test, y_test = origin_data[test_index, 0:-1], origin_data[test_index, -1]
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=10)
        knn.fit(x_train, y_train)
        y_hat = knn.predict(x_test)
        y_train_predict = knn.predict(x_train)
        test_acc_history.append(accuracy_score(y_test, y_hat))
        train_acc_history.append(accuracy_score(y_train, y_train_predict))
    return np.mean(train_acc_history), np.mean(test_acc_history)


n_fold = 10
writer = SummaryWriter(log_dir=f'logs/KNN', flush_secs=2)
log_value = writer.add_scalars
for algorithm in ["ball_tree", "kd_tree", "brute"]:
    for weights in ["uniform", "distance"]:
        for neighbor in range(2, 50, 2):
            train_acc, test_acc = fit_by_knn(n_fold, n_neighbors=neighbor, algorithm=algorithm)
            log_value(f"Accuracy-{algorithm}-{weights}", {"train": train_acc, "test": test_acc}, neighbor)

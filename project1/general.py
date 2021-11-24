# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2021-11-23 22-34
@file: general.py
"""
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.svm import SVC
import pickle

origin_data = np.loadtxt("./kddtrain2021.txt")
sample_len = origin_data.shape[0]
x_train, y_train = origin_data[:, 0:-1], origin_data[:, -1]

test_data = np.loadtxt("./kddtest2021-1.txt")

decision_tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=10)
decision_tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=12)

svm_poly = SVC(decision_function_shape='ovo', kernel="poly", degree=4)
svm_rbf = SVC(decision_function_shape='ovo', kernel="rbf")

mlp_logistic = MLPClassifier(hidden_layer_sizes=(90,), activation="logistic", solver="adam", max_iter=1000)
mlp_tanh = MLPClassifier(hidden_layer_sizes=(90,), activation="tanh", solver="adam", max_iter=1000)
mlp_relu = MLPClassifier(hidden_layer_sizes=(170,), activation="relu", solver="adam", max_iter=1000)

models = [decision_tree_gini, decision_tree_entropy, svm_poly, svm_rbf, mlp_logistic, mlp_tanh, mlp_relu]
predict = np.zeros((test_data.shape[0], len(models)))
for idx, model in enumerate(models):
    model.fit(x_train, y_train)
    model_pkl = pickle.dumps(model)
    with open(f"models/{model}.pkl", "wb") as model_file:
        model_file.write(model_pkl)
    model_predict = model.predict(test_data)
    predict[:, idx] = model_predict

np.savetxt("general_predict.txt", X=predict, fmt="%i")

# 投票法
predict = predict.astype(int)
predict_result = np.zeros((predict.shape[0], 1))
for idx in range(predict.shape[0]):
    cls = np.argmax(np.bincount(predict[idx, :]))
    predict_result[idx, 0] = cls
np.savetxt("predict.txt", X=predict_result, fmt="%i")

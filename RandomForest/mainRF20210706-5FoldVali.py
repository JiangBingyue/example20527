import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.datasets import load_boston

switch=[1,0,0,0,0,0]
switch1=[0]#1按照牌号划分数据。0任意牌号随意抽取测试数据
pltimportance=True

def tunning(model,param_name,param_range,X_train, y_train,visualize=False):
    param_test={}
    if len(param_name)==2 or type(param_range[0])==range:
        for ii in range(len(param_name)):
            param_test[param_name[ii]] = param_range[ii]
    else:
            param_test[param_name[0]] = param_range
    gsearch=GridSearchCV(estimator=model,
                          param_grid=param_test,
                          scoring='neg_mean_squared_error',
                          n_jobs=4,
                          iid=False,
                          cv=5)
    gsearch.fit(X_train, y_train.ravel())
    print(gsearch.best_params_)
    print(gsearch.best_score_)
    if visualize:
        if len(param_name)==2:
            grid_visualization = []
            for grid_pair in gsearch.cv_results_['mean_test_score']:
                grid_visualization.append(grid_pair)
            grid_visualization = np.array(grid_visualization)
            grid_visualization.shape = (len(param_range[0]), len(param_range[1]))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xticks(range(grid_visualization.shape[1]))
            ax.set_xticklabels(param_range[1])
            ax.set_yticks(range(grid_visualization.shape[0]))
            ax.set_yticklabels(param_range[0])
            plt.xlabel(param_name[1])
            plt.ylabel(param_name[0])
            im = ax.imshow(grid_visualization, cmap=plt.cm.hot)
            plt.colorbar(im)
            plt.show()
        else:
            grid_visualization = []
            for grid_pair in gsearch.cv_results_['mean_test_score']:
                grid_visualization.append(grid_pair)
            plt.plot(gsearch.param_grid[param_name[0]],grid_visualization)
            plt.show()

def write_csv(data, name):
    file_name = name
    save = pd.DataFrame(list(data))
    try:
        save.to_csv(file_name)
    except UnicodeEncodeError:
        print("编码错误")

directory=['\data_lattice']
path0='data20210706-dataSet2.xlsx'
X_train=[]
y_train=[]
X_test=[]
y_test=[]
data=np.array(pd.read_csv(os.path.join(os.path.abspath('.\\RandomForest'),'data.csv'),header=None))
len_data=data.shape[0]
idx_data=np.arange(len_data)
kf=KFold(n_splits=5,shuffle=True)
X=data[:,[0,1,2,3,4,5,6]]
Y=data[:,[7]]
# X_train = X[idx_train, :]
#
# y_train = Y[idx_train, :]
# X_test = X[idx_test, :]
# y_test = Y[idx_test, :]
model = RandomForestRegressor(n_estimators=100, max_depth=4,
                              max_features=4, min_samples_leaf=4,
                              min_samples_split=2,
                              oob_score=True, random_state=0)
arrayLossTest = []
arrayLossTrain = []
for train_idx, test_idx in kf.split(idx_data):
    X_train = X[train_idx, :]
    y_train = Y[train_idx, :]
    X_test = X[test_idx, :]
    y_test = Y[test_idx, :]
    model.fit(X_train, y_train.ravel())
    y_test_predict = model.predict(X_test)
    y_train_predict = model.predict(X_train)
    data_train_concat = np.concatenate([y_train, y_train_predict[:, np.newaxis]], axis=1)
    data_test_concat = np.concatenate([y_test, y_test_predict[:, np.newaxis]], axis=1)
    write_csv(data_train_concat, 'pred_train.csv')
    write_csv(data_test_concat, 'pred_test.csv')
    if pltimportance:
        # importances = model.feature_importances_
        # cols=list(range(len(importances)))
        # indices=np.argsort(importances)
        # cols=[cols[x] for x in indices]
        # plt.figure(figsize=(10,6))
        # plt.title('feature importance')
        # plt.barh(range(len(importances)),importances[indices],color='b')
        # plt.yticks(range(len(importances)),cols)
        # plt.xlabel('relative importance')
        # y_test_predict = model.predict(X_test)
        # plt.show()
        # plt.scatter(X_test[:,0],y_test,marker='x', color='blue', s=20)
        # plt.scatter(X_test[:, 0],y_test_predict, marker='x', color='red', s=20)
        # plt.xlabel('strain')
        # plt.ylabel('stress')
        # plt.show()
        data_train_concat = np.concatenate([y_train, y_train_predict[:, np.newaxis]], axis=1)
        data_test_concat = np.concatenate([y_test, y_test_predict[:, np.newaxis]], axis=1)
        write_csv(data_train_concat, 'pred_train.csv')
        write_csv(data_test_concat, 'pred_test.csv')

    arrayLossTest.append(np.mean(np.abs(y_test-y_test_predict[:, np.newaxis])/y_test))
    arrayLossTrain.append(np.mean(np.abs(y_train-y_train_predict[:, np.newaxis])/y_train))

# print('test ERROR mean:%.4f ;std:%.4f' % \
#       (np.array(arrayLossTest).mean(),np.array(arrayLossTest).std()))
# print('train ERROR mean:%.4f ;std:%.4f' % \
#       (np.array(arrayLossTrain).mean(), np.array(arrayLossTrain).std()))
print('%.4f(%.4f)' % (np.array(arrayLossTest).mean(),np.array(arrayLossTest).std()))
print('%.4f(%.4f)' % (np.array(arrayLossTrain).mean(), np.array(arrayLossTrain).std()))



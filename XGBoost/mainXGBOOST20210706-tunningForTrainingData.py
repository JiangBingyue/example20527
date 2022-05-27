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

switch=[0,0,0,0,1,0]
switch1=[0]#1按照牌号划分数据。0任意牌号随意抽取测试数据
pltimportance=True

def tunning(param_name,param_range,visualize=False):
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
                          cv=5)
    gsearch.fit(X_train, y_train)
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
            im = ax.imshow(grid_visualization, cmap=plt.cm.hot,interpolation='nearest')
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

# directory=['\data_lattice']
path0='data.csv'
X_train=[]
y_train=[]
X_test=[]
y_test=[]
data=np.array(pd.read_csv(os.path.join(os.path.abspath('.\\RandomForest'),'data.csv'),header=None))
x = data
idxs = list(np.arange(x.shape[0]))
idx_test = np.random.randint(0, x.shape[0], 15)
idx_train=list(set(idxs).difference(set(idx_test)))
X=x[:,[0,1,2,3,4,5,6]]
Y=x[:,[7]]
X_train = X[idx_train, :]

y_train = Y[idx_train, :]
X_test = X[idx_test, :]
y_test = Y[idx_test, :]
model = xgb.XGBRegressor(learning_rate=0.1,
                         n_estimators=100, max_depth=1,
                         min_child_weight=2, 
                         subsample=0.68,
                         colsample_bytree=0.72, gamma=0.,
                         reg_alpha=0.001,
                         objective='reg:gamma')
if pltimportance:
    model.fit(X_train, y_train.ravel())
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
    y_test_predict = model.predict(X_test)
    y_train_predict = model.predict(X_train)
    data_train_concat=np.concatenate([y_train,y_train_predict[:,np.newaxis]],axis=1)
    data_test_concat = np.concatenate([y_test, y_test_predict[:, np.newaxis]], axis=1)
    write_csv(data_train_concat, 'pred_train.csv')
    write_csv(data_test_concat, 'pred_test.csv')


    # plt.scatter(X_test[:,0],y_test,marker='x', color='blue', s=20)
    # plt.scatter(X_test[:, 0],y_test_predict, marker='x', color='red', s=20)
    # plt.xlabel('strain')
    # plt.ylabel('stress')
    # plt.show()




if switch[0]:
    param_name0=['n_estimators']
    param_range0=[range(10,200,2)]
    tunning(param_name0,param_range0,visualize=True)

if switch[1]:
    # max_depth_range=range(3,10,2)
    # min_chid_weight=range(1,6,2)
    param_name1=['max_depth','min_child_weight']
    param_range1=[range(1,5,1),range(1,6,1)]
    tunning(param_name1, param_range1, visualize=True)

if switch[2]:
    param_name2=['gamma']
    param_range2=[x/10.0 for x in range(0,5)]
    tunning(param_name2,param_range2,visualize=True)

if switch[3]:
    param_name3 = ['subsample','colsample_bytree']
    param_range3=[[i / 100.0 for i in range(10,100,2)],
                  [i / 100.0 for i in range(10,100,2)]]
    tunning(param_name3, param_range3, visualize=True)

if switch[4]:
    param_name4=['reg_alpha']
    param_range4=[2e-5,2.5e-5,3e-5,3.5e-5,4e-5,1e-4,1e-3,1e-2,0.1]
    tunning(param_name4, param_range4, visualize=True)

if switch[5]:
    param_name5 = ['learning_rate']
    param_range5 = [i/10000 for i in range(100,9000,100)]
    tunning(param_name5, param_range5, visualize=True)

# # 对测试集进行预测
# ans = model.predict(X_test)
#
# # 显示重要特征
# plot_importance(model)
# plt.show()

#导入数据集
# boston = load_boston()
# X, y = boston.data, boston.target
#
# # Xgboost训练过程
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 20:58:50 2022

@author: My
"""
import datetime
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

data1 = pd.read_csv('data_temps1.csv',encoding='utf-8') 
data1

data1.describe()

years = data1['年']
months = data1['月']
days = data1['日']

# data1_dates=[]
# for 年,月,日 in zip(years,months,days):
#     data1_dates+=[str(int(年))+'-' + str(int(月)) + '-' + str(int(日))] 
data1_dates = [str(int(年)) + '-' + str(int(月)) + '-' + str(int(日)) for 年, 月, 日 in zip(years, months, days)]#列表这样写方便
    
#转化为计算机可识别的时间类型
# 第二种写法
# for j in range(len(data1_dates)):
#     data1_dates[j]=datetime.datetime.strptime(data1_dates[j],'%Y-%m-%d')
data1_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in data1_dates] 

#把分类变量虚拟变量化
data1 = pd.get_dummies(data1)

data1_label = data1['当天最高温度']
data1_feature= data1.drop('当天最高温度', axis = 1)
data1_lable = np.array(data1_label) #数组形式才能分train和test
data1_feature = np.array(data1_feature)

data1_train_feature,data1_test_feature,data1_train_label,data1_test_label=train_test_split(data1_feature,data1_label,
                                                                                           test_size=0.25,random_state=0)
#第一个训练集和测试集就出来了
#%%
#同样的方法把第二个训练集和测试集弄出来
data2=pd.read_csv('data_temps2.csv')

#还是一样把年月日转换
year1=data2['年']
month1=data2['月']
day1=data2['日']
data2_date=[str(int(年))+'-'+str(int(月))+'-'+str(int(日)) for 年,月,日 in zip(year1,month1,day1)]
data2_date=[datetime.datetime.strptime(date,'%Y-%m-%d') for date in data2_date]
data2=pd.get_dummies(data2)

data2_label=data2['当天最高温度']
data2_feature=data2.drop('当天最高温度',axis=1)
list_=list(data2_feature.columns)#后续中找不同

data2_label=np.array(data2_label)
data2_feature=np.array(data2_feature)
#第二个训练集
data2_train_feature,data2_test_feature,data2_train_label,data2_test_label=train_test_split(data2_feature,data2_label,
                                                                                           test_size=0.25,random_state=0)
#算出两个训练集中不同的feature
difference=[list_.index(m) for m in list_ if m not in ['前一天风速','前一天降水','前一天积雪深度']]  #除开三个指标的索引，因为后续处理一致性测试集

#%%
#第三个数据集，因为第二数据集比第一个数据集在样本和feature上都多，为了验证样本和feature带来的预测精度
#此时获得与数据集一样的feature，但是样本与数据集二相同
data3=data2.drop(['前一天风速','前一天降水','前一天积雪深度'],axis=1)
data3_label=data3['当天最高温度']
data3_feature=data3.drop('当天最高温度',axis=1)
data3_label=np.array(data3_label)
data3_feature=np.array(data3_feature)
#第三个训练集
data3_train_feature,data3_test_feature,data3_train_label,data3_test_label=train_test_split(data3_feature,data3_label,
                                                                                           test_size=0.25,random_state=0)
#%%
#开始将三个训练集来拟合模型
model=RandomForestRegressor(n_estimators=100,random_state=0)#自己先试着弄100个决策树，看下效果
#为了比较三个样本，这么说是三个训练集,将测试集都弄成一样的
data1_test_feature=data2_test_feature[:,difference]
data1_test_label=data2_test_label

data3_test_feature=data2_test_feature[:,difference]
data3_test_label=data2_test_label

#预测精准度方程
def precise(predict,initial):
    error=predict-initial
    error_mean=100*np.mean(error/initial)
    return (100-error_mean)

#第一个数据集建立的模型
model1=model.fit(data1_train_feature,data1_train_label)
predict1=model1.predict(data1_test_feature)

print(f'第一个数据集准确度{precise(predict1,data1_test_label)}%')

#第二个数据集建立的模型
model2=model.fit(data2_train_feature,data2_train_label)
predict2=model2.predict(data2_test_feature)

print(f'第二个数据集准确度：{precise(predict2,data2_test_label)}%')

#第三个数据集建立的模型
model3=model.fit(data3_train_feature,data3_train_label)
predict3=model3.predict(data3_test_feature)

print(f'第三个数据集准确度：{precise(predict3,data3_test_label)}%')

comparison={'训练集':['253个样本量+14个指标','1635个样本量+17个指标','1635个样本量+14个指标'],
            '准确度%':[96.8830,99.3531,99.1945]}
comparison=pd.DataFrame(comparison)
#所以选择第二个训练集

#%%
#调整超参数
#六个参数调整RandomSearchCV中的 param_distributions   
#1.max_features变量选择方式[auto,sqrt] 2.决策树的个数n_estimators 3.max_depth树的最大深度[5,10]
#4.训练集中样本采样方式[True,False],是有放回的抽，还是不放回的抽  5.min_samples_split一个结点必须要包含至少min_samples_split个训练样本，这个结点才允许被分裂
#6.min_samples_leaf一个结点在分支后的每个子结点都必须包含至少min_samples_leaf个训练样本，否则分支不会发生
max_features=['auto','sqrt']
n_estimators=[int(x) for x in np.linspace(20,200,10)]
max_depth=[10,20,None]
bootstrap=[True,False]
min_samples_split=[2,5,10]
min_samples_leaf=[2,4,6]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf=RandomForestRegressor()
#用随机搜索来优化RandomForestRegressor
rf_random=RandomizedSearchCV(estimator=rf, param_distributions=random_grid,n_iter=10,scoring='neg_mean_absolute_error',cv=3,random_state=0)

rf_random.fit(data2_train_feature,data2_train_label)

rf_random.best_params_#取得最优参数,在这个基础上调整网格搜索

def accuracy(model,data_test_feature,data_test_label):
    predict=model.predict(data_test_feature)
    error=100*np.mean((predict-data_test_label)/data_test_label)
    precise=100-error
    print('随机搜索的调整：')
    print(f'平均绝对误差error:{error}')
    print(f'准确度为：{precise}%')

#获得最优模型
rf_estimator=rf_random.best_estimator_
precise_randomsearchcv=accuracy(rf_estimator, data2_test_feature, data2_test_label)
#%%
#再用网格搜索来优化(GridSearchCV)
#同样也是6个参数调整GridSearchCV中的param_grid,保留刚才在随机搜索中的三个参数：max_feature,max_depth,bootstrap;
# =============================================================================
# #调整n_estimator，min_samples_split,min_samples_leaf的值,之前在随机搜索中的最优值分别是100,10,2
# #增大这三个值
# =============================================================================
n_estimators=[110,120,130]
min_samples_split=[11,12,13]
min_samples_leaf=[3,4,5]
max_features=['sqrt']
max_depth=[10]
bootstrap=[False]

param_first_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


grid_first_search=GridSearchCV(estimator=rf, param_grid=param_first_grid,
                               scoring='neg_mean_absolute_error',cv=3)

grid_first_search.fit(data2_train_feature,data2_train_label)
grid_first_search.best_params_
print('第一次网格搜索的调整')
best_grid_search=grid_first_search.best_estimator_
accuracy(best_grid_search, data2_test_feature, data2_test_label)

# =============================================================================
# #减小这三个值
# =============================================================================
n_estimators=[70,80,90]
min_samples_split=[7,8,9]
min_samples_leaf=[1,2]
max_features=['sqrt']
max_depth=[10]
bootstrap=[False]

param_first_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


grid_second_search=GridSearchCV(estimator=rf, param_grid=param_first_grid,
                               scoring='neg_mean_absolute_error',cv=3)

grid_second_search.fit(data2_train_feature,data2_train_label)
grid_second_search.best_params_
print('第二次网格搜索的调整')
best_grid_search=grid_second_search.best_estimator_
accuracy(best_grid_search, data2_test_feature, data2_test_label)
# =============================================================================
# #画出最优模型下的第一课决策树
# =============================================================================
import pydotplus
import graphviz
from IPython.display import Image
from sklearn import tree
dot_data=tree.export_graphviz(
    best_grid_search.estimators_[0],
    out_file=None,
    feature_names=list_,
    filled=True,
    impurity=True,
    rounded=True,
    special_characters="utf-8"
 )
graph=graphviz.Source(dot_data.replace('helvetica', '"Microsoft YaHei"'), encoding='utf-8')
graph.view()
# =============================================================================
# 这里运行
# =============================================================================

# =============================================================================
# #然后就一直调整这三个系数，直到准确度最优
# =============================================================================
overall={'方式':['普通的随机森林回归','随机搜索调整','第一次网格搜索调整','第二次网格搜索调整'],
         '准确度%':[99.3531,99.3547,99.3657,99.3751],
         '决策树的个数选择':[100,100,110,90]}
overall=pd.DataFrame(overall)
#说明第二次网格搜索的准确度最高，作为最优模型，来预测test

















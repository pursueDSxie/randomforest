# randomforest
随机森林，该模型用于预测气候温度

<p>问题重述：预测当日温度,运用RandomForestRegressor(随机森林回归)<o/>
<p>1.确定最优训练集：<o/>
第一个训练集拥有253个样本+14个指标
第二个训练集拥有1635个样本+17个指标
第三个训练集拥有1635个样本+14个指标
最终确定为第二个训练集预测精确度最高
	
2.利用第二个训练集，调整随机森林模型超参数（6大基本参数，在图片里面）
以下两大方法调整
运用from sklearn.model_selection import RandomSearchCV
运用from sklearn.model_selection import GridSearchCV
	
不断调整参数，比较预测准确度，最终确定最优模型。

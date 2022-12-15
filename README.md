# randomforest
随机森林，该模型用于预测气候温度

<p>问题重述：预测当日温度,运用RandomForestRegressor(随机森林回归)<o/>
<p>1.确定最优训练集：<o/>
<p>第一个训练集拥有253个样本+14个指标<o/>
<p>第二个训练集拥有1635个样本+17个指标<o/>
<p>第三个训练集拥有1635个样本+14个指标<o/>
<p>最终确定为第二个训练集预测精确度最高<o/>
	
<p>2.利用第二个训练集，调整随机森林模型超参数<o/>
<p>以下两大方法调整<o/>
<p>运用from sklearn.model_selection import RandomSearchCV<o/>
<p>运用from sklearn.model_selection import GridSearchCV<o/>
	
<p>不断调整参数，比较预测准确度，最终确定最优模型。<o/>

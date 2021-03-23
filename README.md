### 任务：

对简历数据进行二分类，判断其是否属于数字经济人才.

### 准备工作:
    
需要下载BERT中文训练文件[chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

Tensorflow版本为1.14（暂未配置gpu训练环境）

### 模型：

BERT作为特征提取, 多种模型进行二分类, 目前有svm,random forest,naive bayes,dnn

### 使用说明：
	
	训练数据与测试样例在data下，输出的测试结果保存在result下，执行model_train.py即可。
	为了便于在excel中查看数据，统一使用gbk编码

### 参考：

<https://www.cnblogs.com/jclian91/p/12301056.html>
	
<https://github.com/percent4/bert_doc_binary_classification>

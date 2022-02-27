## aspect_level_sentiment_analysis

### 1、环境：python3.5.4   TensorFlow 1.4.0
### 2、说明：数据集太大了，处理后的数据只存放了验证集分词的txt在Dataset文件夹中。训练后的模型文件存放在model文件夹中，因为较多，故每个细粒度的分类模型（20个细粒度分类器）中的表现好的那个做代表


####     面向细粒度的情感分析，2018年度AI Challenger全球AI挑战赛
####     本项目内容为餐厅评论领域的细粒度情感分析，涉及20个细粒度要素的情感倾向分类任务
####     使用的是面向短评论的细粒度情感分析方法Bi-LSTMSA（Bi-directional Long Short-term Memory and Self-Attention），将细粒度情感分析任务分为输入向量处理模块、Bi-LSTM网络模块和Self-Attention机制模块。输入向量处理模块引入情感词库参与分词并借助词向量技术训练得到词语向量；Bi-LSTM网络模块采用双向长短记忆网络提取时间序列下句子信息；Self-Attention机制模块根据相似度原则，计算词语间依赖关系信息并添加到表达词语的向量中，用于弥补梯度损失。
####     结果表明，Bi-LSTMSA相比基础LSTM网络和单纯Self-Attention机制两种基准网络，可以有效提高细粒度情感分类模型的准确率、各类别的F1-score和降低损失值。在2018 AI Challenger情感分析赛道数据集上，Bi-LSTMSA在75%的细粒度要素上达到0.8以上的准确率，最高准确率为0.9722，在“未提及”和“积极”情感上的F1-score大部分表现较好。

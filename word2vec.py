# 词向量模型的训练
from gensim.models import word2vec
import datetime
class wordItor():
    def __init__(self, path):
        self.path = path
        self.content = self.readData()

    def __iter__(self):
        for i in range(len(self.content)):
            # 去除标点
            text = self.content[i]
            #返回分词结果，去掉末尾的\n
            yield text.split(" ")[:-1]

    def readData(self):
        file = open(self.path, 'r', encoding='utf-8-sig')
        content = []
        for line in file:
            line = line.strip('\n')
            content.append(line)
        return content


# 读取train中content部分，经过预处理以及分词后，返回一个word的迭代器
path="./dataset/all_content.txt"
sen = wordItor(path)
# 传入迭代器，开始训练
size = [20,40,60,80,100,120,140,160,180,200,220, 240,260,280,300, 320,340,360]
for i in size:
    #进行训练：维度为i，窗口大小为10，skip-gram模型，词频小于3的去掉
    model = word2vec.Word2Vec(sentences=sen, size=i,window=10,sg=1,min_count=3,iter=10,negative=15)
    model.save('./model/w2v/w2v_model_' + str(i) + '_size.model')
    print("维度为："+str(i)+"的词向量模型训练OK")

import pandas as pd
import re
from langconv import *

#句子去掉非中文字符
def remove_unchinese(sentence):
    return re.sub(u'[^\u4e00-\u9fa5]', '', sentence)

################停用词库处理###############
#合并四个停用词库
stopwords=[]
stopwordtxt=["/baidu_stopwords.txt","/china_stopwords.txt","/hagongda_stopwords.txt","/sichuanuniversity_stopwords.txt"]
for i in range(len(stopwordtxt)):
    path="./stopwords%s"%(stopwordtxt[i])
    with open(path,"r",encoding='UTF-8') as file:
        for line in file.readlines():
            line=line.strip('\n')
            stopwords.append(line)
print("原四个停用词库总长度："+str(len(stopwords)))
stopwords_set=set(stopwords)
print("去掉相同词后停用词库的总长度："+str(len(stopwords_set)))

#有关评价和否定的词语
sentimentword=[]
txtlist=["/程度级别词语219.txt","/负面评价词语3116.txt","/负面情感词语1254.txt","/正面评价词语3730.txt","/正面情感词语836.txt","/否定词语1428.txt","/味觉词库182.txt"]
for i in range(len(txtlist)):
    path="./sentiment_dictionary%s"%(txtlist[i])
    with open(path,"r",encoding='utf-8') as file:
        for line in file.readlines():
            line=line.strip('\n')
            sentimentword.append(line)
print("原情感词典总长度："+str(len(sentimentword)))
sentimentword_set=set(sentimentword)
print("去掉相同词后情感词典的总长度："+str(len(sentimentword_set)))

#去掉评价词语之后的停用词
final_stopwords=list(stopwords_set-sentimentword_set)
print("去掉停用词中包含的情感词后的停用词库总长度："+str(len(final_stopwords)))

#最终的停用词库，去掉非中文字符，并写入TXT
real_final_stopwords=[]
with open("./stopwords/final_stopword.txt","w",encoding='utf-8') as final_stopwordfile:
    for line in final_stopwords:
        line=remove_unchinese(line)
        if(line!=''):
            final_stopwordfile.write(line+'\n')
            real_final_stopwords.append(line)
final_stopwordfile.close()
print("去掉非中文字符后停用词的总个数："+str(len(real_final_stopwords)))

#最终的情感词库
final_sentimentword=list(sentimentword_set)
with open("./sentiment_dictionary/final_dictionary.txt","w",encoding='utf-8') as final_dicfile:
    for line in final_sentimentword:
        line=remove_unchinese(line)
        if(line!=''):
            final_dicfile.write(line+'\n')
final_dicfile.close()
print("总情感词库词数："+str(len(final_sentimentword)))
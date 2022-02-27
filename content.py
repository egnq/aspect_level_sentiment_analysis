
import pandas as pd
import re
from langconv import *
import jieba
from time import time
#句子去掉非中文字符
def remove_unchinese(sentence):
    return re.sub(u'[^\u4e00-\u9fa5]', '', sentence)

#句子繁体字转为简体字
def Traditional2Simplified(sentence):
    sentence=Converter('zh-hans').convert(sentence)
    sentence.encode('utf-8')
    return sentence

#结巴中文分词，返回list
def jiebadivide(sentence):
    return jieba.lcut(sentence)

#判断是否是停用词
def in_stopword(_word,_stopwords):
    if _word not in _stopwords:
        return False
    return True

#合并处理操作
def textHandler(sentence,real_final_stopwords):
    sentence=Traditional2Simplified(sentence)
    sentence=remove_unchinese(sentence)
    words = jieba.lcut(sentence)
    outstr = ''
    _stopwords = real_final_stopwords
    for word in words:
        if in_stopword(word,_stopwords) == False:
            outstr += word
            outstr += " "
    outstr = outstr + '\n'
    return outstr

#读取停用词库
def read_stopword(path):
    stopword = []
    with open(path,"r",encoding='utf-8') as file:
        for line in file.readlines():
            line=line.strip('\n')
            stopword.append(line)
    return stopword
#对用户的文本进行分词写入
def div_content_writeinto_txt(stopword_path='./stopwords/final_stopword.txt',sentences=None):
    method = int(time())
    if sentences==None:
        return
    real_final_stopwords = read_stopword(stopword_path)
    jieba.load_userdict("./sentiment_dictionary/final_dictionary.txt")
    path='./dataset/user_content/%s_content.txt' % (method)
    file = open(path, 'w', encoding='utf-8')
    for con in sentences:
        deal_contant = textHandler(con, real_final_stopwords)
        file.write(deal_contant)
    file.close()
    return path
#按行读取txt
def read_txt(path):
    content = []
    try:
        with open(path,"r",encoding='utf-8-sig') as file:
            for line in file.readlines():
                line=line.strip('\n')
                content.append(line)
    except:
        with open(path,"r") as file:
            for line in file.readlines():
                line=line.strip('\n')
                content.append(line)
    return content
#读取已经分好词的TXT文件
def read_divcontent_file(path):
    add_each_sentence_length=[]
    final_content=[]
    content=[]
    with open(path,"r",encoding='utf-8-sig') as file:
        for line in file.readlines():
            line=line.strip('\n')
            content.append(line)
    num = 0
    add_each_sentence_length.append(num)
    each_length=[]
    for i in range(len(content)):
        j=0
        text = content[i]
        for word in text.split(" ")[:-1]:
            final_content.append(word)
            num+=1
            j+=1
        add_each_sentence_length.append(num)
        each_length.append(j)
    return final_content,add_each_sentence_length,each_length

#########################语料库预处理########################
#繁体字转简体字
#去掉非中文字符
#jieba分词
#去掉停用词
########train训练集、test测试集、validation验证集########

def deal_data(stopword_path='./stopwords/final_stopword.txt',method=None,io=None,inputs=None,style=None):
    real_final_stopwords = read_stopword(stopword_path)
    if method in ['train','test','validation']:
        path=io
        dataframe = pd.read_csv(path)
        content = dataframe['content']
        jieba.load_userdict("./sentiment_dictionary/final_dictionary.txt")
        file = open('./dataset/%s_content.txt' % (method), 'w', encoding='utf-8')
        for index in range(len(content)):
            deal_contant = textHandler(content[index], real_final_stopwords)
            file.write(deal_contant)
            if index % 2000 == 0:
                print(index)
        file.close()
        return
    else:
        return





if __name__ == '__main__':
    io=[r"./dataset/sentiment_analysis_trainingset.csv",r"./dataset/sentiment_analysis_testa.csv",r"./dataset/sentiment_analysis_validationset.csv"]
    method=['train','test','validation']
    for i in range(3):
        path=io[i]
        methods=method[i]
        deal_data(io=path,method=methods)






















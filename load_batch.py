# python3
# -*- coding:utf-8 -*-
import linecache
from gensim.models import Word2Vec
import numpy as np


class Load_batch:


    def __init__(self, method='train', label=-1, batchSize=200,
                 maxSeqLength=150, numDimensions=200):
        label_str = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
                 'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience',
                 'service_serving_speed',
                 'price_level', 'price_cost_effective', 'price_discount',
                 'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
                 'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
                 'others_overall_experience', 'others_willing_to_consume_again']
        linecache.clearcache()  # 清除缓存，防止读取脏数据
        self.numDimensions = numDimensions
        self.maxSeqLength = maxSeqLength
        self.batchSize = batchSize
        if method is 'train':
            self.length = 104999
        else:
            self.length = 14999
        self.content_file_path = './dataset/%s_content.txt' % (method)
        self.label_file_path = './dataset/label/%s_label_%s.txt' % (method,label_str[label])
        self.model_path = './model/w2v/w2v_model_%s_size.model'% (numDimensions)
        self.model = Word2Vec.load(self.model_path)
        self.seg_itor = self.get_seg_batch()
        self.label_itor = self.get_label_batch()

    def next(self):
        segMatrix, segLen = next(self.seg_itor)
        labelVec = next(self.label_itor)
        #（batchSize,maxSeqLength,numDimensions）,(batchsize), [batchSize, 4]
        return segMatrix, segLen, labelVec

#获取得到content的batch
    def get_seg_batch(self):
        maxSeqLength = self.maxSeqLength#句子最大长度
        numDimensions = self.numDimensions#词向量维数
        file_path = self.content_file_path
        idx = 0
        while True:
            start = idx
            idx = (idx+self.batchSize)% self.length#超过数据集最大长度从头开始
            segMatrix = None#存储一个batch的词向量
            segLen = []#存储一个batch中每个句子的长度
            for i in range(self.batchSize):
                ######对于每个句子######
                line = None#存储一个句子的词向量
                currSeqLength = 0
                #对第numline行进行操作，从1开始算起
                numline=start + i + 1
                if numline>self.length:
                    numline=numline%self.length#超过数据集最大长度从头开始
                #读取第numline行，以‘ ’为分解
                for word in linecache.getline(file_path, numline).split():
                    try:
                        #词向量中查找该词语
                        vec = self.model[word].reshape(1, numDimensions)
                    except KeyError:
                        # 未登录词汇随机
                        vec=np.random.normal(0,0.08,numDimensions).reshape(1,numDimensions)
                    if line is None:
                        line = vec
                    else:
                        line = np.append(line, vec, axis=0)

                    currSeqLength += 1
                    if currSeqLength >= maxSeqLength:
                        break
                segLen.append(currSeqLength)
                #句子太短时，补0
                while currSeqLength < maxSeqLength:
                    line = np.append(line, np.zeros((1, numDimensions)), axis=0)
                    currSeqLength += 1
                line = line.reshape(1, maxSeqLength, numDimensions)
                if segMatrix is None:
                    segMatrix = line
                else:
                    segMatrix = np.append(segMatrix, line, axis=0)
            #返回一个batch的词向量三维矩阵（batchSize,maxSeqLength,numDimensions）和各个句子的实际长度(batchsize)
            yield segMatrix, np.asarray(segLen)

    def get_label_batch(self):
        batchSize = self.batchSize
        file_path = self.label_file_path
        idx = 0
        while True:
            start = idx
            idx = (idx + self.batchSize) % self.length
            label = None#存储一个batch的label的one-hot encoding形式
            for i in range(batchSize):
                #[未提及，消极，中性，积极]
                line = [0, 0, 0, 0]
                numline = start + i + 1
                if numline > self.length:
                    numline = numline % self.length
                #对第numline的label进行转化为one-hot形式
                num = int(linecache.getline(file_path, numline))
                num += 2
                line[num] += 1
                line = np.asarray(line).reshape(1, 4)

                if label is None:
                    label = line
                else:
                    label = np.append(label, line, axis=0)
            # [batchSize, 4]
            yield label

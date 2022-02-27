import tensorflow as tf
import os
from time import time
from content import read_divcontent_file,div_content_writeinto_txt,read_txt
from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定中文字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def plot_with_labels(low_dim_embs, labels, filename):   # 绘制词向量图
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(10, 10))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)	# 画点，对应low_dim_embs中每个词向量
        plt.annotate(label,	# 显示每个点对应哪个单词
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
#获取输入为TXT文档或者已经明确句子个数每一行为一个句子的文本的分词结果，
# 返回divide_sentences,所有词写在一个列表里,each_length[0,10,20,50,..]
def clean_data(stopword_path='./stopwords/final_stopword.txt',file_path=None,sentences=None):
    true_sentences=sentences
    if file_path == None and sentences==None:
        return None,None,None
    #如果以TXT文件输入
    if file_path!=None and sentences==None:
        true_sentences=read_txt(path=file_path)
    path=div_content_writeinto_txt(sentences=true_sentences)
    #返回结果为一维的所有句子加起来的词语、每个句子的长度和前面句子长度的和
    return read_divcontent_file(path=path)
#获取输入对应的词向量,返回（句子数,maxSeqLength,numDimensions）和各个句子的实际长度(句子数)
def get_w2v(divide_sentences,each_length):
    true_sentences,true_length=divide_sentences,each_length
    num_sentences=len(true_length)-1

    segMatrix = None  # 存储一个batch的词向量
    segLen = []  # 存储一个batch中每个句子的长度
    numDimensions=300
    maxSeqLength=150
    model_path = './model/w2v/w2v_model_%s_size.model' % (numDimensions)
    model = Word2Vec.load(model_path)
    for i in range(num_sentences):
        ######对于每个句子######
        line = None  # 存储一个句子的词向量
        currSeqLength = 0
        start=true_length[i]
        finish=true_length[i+1]

        for word in true_sentences[start:finish]:
            try:
                # 词向量中查找该词语
                vec = model[word].reshape(1, numDimensions)
            except KeyError:
                # 未登录词汇随机
                vec = np.random.normal(0, 0.08, numDimensions).reshape(1, numDimensions)
            if line is None:
                line = vec
            else:
                line = np.append(line, vec, axis=0)

            currSeqLength += 1
            if currSeqLength >= maxSeqLength:
                break
        segLen.append(currSeqLength)
        # 句子太短时，补0
        while currSeqLength < maxSeqLength:
            line = np.append(line, np.zeros((1, numDimensions)), axis=0)
            currSeqLength += 1
        line = line.reshape(1, maxSeqLength, numDimensions)
        if segMatrix is None:
            segMatrix = line
        else:
            segMatrix = np.append(segMatrix, line, axis=0)
    # 返回一个batch的词向量三维矩阵（batchSize,maxSeqLength,numDimensions）和各个句子的实际长度(batchsize)
    return segMatrix, np.asarray(segLen)
#绘制词向量矩阵对应的分布,最多画100个词，返回分布图的路径
def plot_w2v_100(divide_sentences):

    divide_sentences=np.array(divide_sentences)
    length=len(divide_sentences)
    word_index=[]
    word_vecs = []
    numDimensions = 300
    model_path = './model/w2v/w2v_model_%s_size.model' % (numDimensions)
    model = Word2Vec.load(model_path)
    true_length=0
    if length > 100:
        j = 0
        while True:
            i = np.random.randint(0, length)
            if i in word_index:
                continue
            else:
                try:
                    model[divide_sentences[i]]
                    word_index.append(i)
                    j += 1
                except KeyError as e:
                    a=1
            if j == 100:
                true_length=100
                break
    else:
        for i in range(length):
            try:
                model[divide_sentences[i]]
                word_index.append(i)
                true_length+=1
            except KeyError as e:
                a = 1
    words = divide_sentences[word_index]
    for w in words:
        vec = model[w]
        word_vecs.append(vec)
    if len(words)<=1:
        return None
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, method='exact')
    low_dim_embs = tsne.fit_transform(word_vecs[:])
    labels = [words[i] for i in range(true_length)]
    method = int(time())
    path='./dataset/user_content/%s_w2v.png' % (method)
    plot_with_labels(low_dim_embs=low_dim_embs,labels=labels,filename=path)
    return path
def test(the_labels,model_names,egMatrix0,each_segLen0):
    model_path = './model/' + model_names + '/label_' + str(the_labels) + '/tensorboard_graph/'
    model_file_name=''
    path = os.listdir(model_path)
    for file in path:
        if '.meta' in file:
            model_file_name=file

    if model_file_name=='':
        raise EOFError('meta文件不存在')

    saver = tf.train.import_meta_graph(model_path + '/' + model_file_name)
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = sess.graph
    input_data = graph.get_tensor_by_name('Inputs/input_data:0')
    input_length = graph.get_tensor_by_name('Inputs/input_length:0')
    labels = graph.get_tensor_by_name('Inputs/labels:0')
    prediction = graph.get_tensor_by_name('Prediction/ArgMax:0')
    attention_output=[]
    if model_names in ["BiLSTM_selfattention","selfattention"]:
        attention_output=graph.get_tensor_by_name('Prediction/Reshape:0')
    else:
        attention_output=None

    data_vector=egMatrix0
    segLen=each_segLen0
    numsentences=len(segLen)
    attentionSize = 128
    maxSeqLength = 150

    all_predict=None
    all_attention=None
    first=0
    end=200
    stept=int(numsentences/200)+1
    #每200个一个batch
    for i in range(stept):
        data=None
        lens=None
        if end>=numsentences:
            data = data_vector[first:numsentences, :, :]
            lens = segLen[first:numsentences]
        else:
            data=data_vector[first:end,:,:]
            lens=segLen[first:end]

        feed_dict = {input_data: data, input_length: lens}
        predict = sess.run(prediction, feed_dict=feed_dict)
        #如果是带有attention的模型
        if attention_output != None:
            attention_vec = sess.run(attention_output, feed_dict=feed_dict)
            attention_vec=np.array(attention_vec)
            attention_vec=attention_vec.reshape(-1,maxSeqLength,attentionSize)
            #用每个词语的向量表示中最大的值作为该词语的attention值
            attention_max=np.max(attention_vec,axis=2)
            if all_attention is None:
                all_attention = attention_max
            else:
                all_attention=np.vstack([all_attention, attention_max])
        else:
            all_attention = None

        if all_predict is None:
            all_predict=np.array(predict)
        else:
            all_predict=np.hstack([all_predict, np.array(predict)])
        first=end
        end+=200
    #[numseq],[numseq,maxseqlength]
    return (all_predict-2),all_attention

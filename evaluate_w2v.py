from gensim.models import KeyedVectors,word2vec,Word2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl

numDimensions=[20,40,60,80,100,120,140,160,180,200,220, 240,260,280,300, 320,340,360]
w2v_model_list=[]
for i in numDimensions:
    model_path = './model/w2v/w2v_model_%s_size.model'% (i)
    model = Word2Vec.load(model_path)
    w2v_model_list.append(model)
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

def load_bin_vec(w2v_model_list,i):
    print('读取词向量文件中......')
    word_vecs = []
    words = []
    model = w2v_model_list[i]
    key=model.wv.similar_by_word('满分', topn=30)
    key1 = model.wv.similar_by_word('极差', topn=30)
    for word in key:
        embedding = model[word[0]]
        word_vecs.append(embedding)
        words.append(word[0])
    for word in key1:
        embedding = model[word[0]]
        word_vecs.append(embedding)
        words.append(word[0])
    return words, word_vecs

if __name__ == '__main__':
    for i in range(18):
        try:
            words, vectors = load_bin_vec(w2v_model_list,i)	# 载入word2vec词向量
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
            low_dim_embs = tsne.fit_transform(vectors[:]) # 需要显示的词向量，一般比原有词向量文件中的词向量个数少，不然点太多，显示效果不好
            labels = [words[i] for i in range(60)] # 要显示的点对应的单词列表
            plot_with_labels(low_dim_embs, labels, './w2vpng/%s w2v.png'%(i))
        except ImportError as ex:
            print(ex)





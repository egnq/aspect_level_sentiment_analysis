import tensorflow as tf
import os

class Load_model:

    def __init__(self, label=-1,numDimensions=100,model_name="BiLSTM_selfattention"):
        label_str = ['location_traffic_convenience', 'location_distance_from_business_district',
                     'location_easy_to_find',
                     'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience',
                     'service_serving_speed',
                     'price_level', 'price_cost_effective', 'price_discount',
                     'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
                     'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
                     'others_overall_experience', 'others_willing_to_consume_again']
        # 模型的参数
        self.lstmUnits = 64
        self.attentionSize =128
        self.dropOutRate = 1.0
        self.learn_rate = 0.0003
        self.numClasses = 4
        self.nhead=8
        # 数据的参数
        self.maxSeqLength = 150
        self.numDimensions = numDimensions
        self.label =label_str[label]
        self.modelid="tensorboard_graph"
        self.model_path = './model/'+model_name+'/label_' + str(self.label) + '/'+ str(self.modelid) + '/'

        if os.path.exists('./model/') is False:
            os.mkdir('./model/')
        if os.path.exists('./model/'+model_name+'/') is False:
            os.mkdir('./model/'+model_name+'/')
        if os.path.exists('./model/'+model_name+'/label_' + str(self.label) + '/') is False:
            os.mkdir('./model/'+model_name+'/label_' + str(self.label) + '/')
        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)

        if model_name=="BiLSTM_selfattention":
            self.graph = self.init_BiLSTM_selfattention_graph()
        else:
            if model_name=="LSTM":
                self.graph=self.init_LSTM_graph()
            else:
                self.graph=self.init_selfattention_graph()

    #生成self-attention的query，key，value
    def attention_QKV(self,inputs, attentionSize, bias=True):
        #inputs shape [[batchsize,maxseqlength,2*lstmunits]
        input_size = int(inputs.shape[-1])  # 只操作最后一个
        W = tf.Variable(
            tf.random_uniform([input_size, attentionSize], -0.05, 0.05))  # shape(in_size,out_size),range(-0.05,0.05)
        if bias:
            b = tf.Variable(tf.random_uniform([attentionSize], -0.05, 0.05))  # shape(out_size),range(-0.05,0.05)
        else:
            b = 0
        outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b  # inputs.shape(*,in_size) * W + b
        outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [attentionSize]], 0))
        #output shape:[batchsize,maxseqlength,attentionsize]
        return outputs
    def F1_score(self,precision,recall):
        a=tf.multiply(precision,recall)
        b=2*a
        c=tf.add(precision,recall)
        d=tf.divide(b,c)
        return d

    def init_BiLSTM_selfattention_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('Inputs'):
                labels = tf.placeholder(tf.float32, [None, self.numClasses], name='labels')
                input_data = tf.placeholder(tf.float32, [None, self.maxSeqLength, self.numDimensions],
                                            name='input_data')
                input_length = tf.placeholder(tf.int32, [None], name='input_length')

            with tf.variable_scope('BiLSTM'):
                encoder_fw = tf.nn.rnn_cell.BasicLSTMCell(self.lstmUnits)
                encoder_fw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_fw,
                                                           output_keep_prob=self.dropOutRate)

                encoder_bw = tf.nn.rnn_cell.BasicLSTMCell(self.lstmUnits)
                encoder_bw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_bw,
                                                           output_keep_prob=self.dropOutRate)

                (
                    (encoder_fw_output, encoder_bw_output),
                    (encoder_fw_state, encoder_bw_state)
                ) = (
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw,
                                                    cell_bw=encoder_bw,
                                                    inputs=input_data,
                                                    sequence_length=input_length,
                                                    dtype=tf.float32)
                )
                #output shape：[batchsize,maxseqlength,2*lstmunits]
                output = tf.concat((encoder_fw_output, encoder_bw_output), 2)

            with tf.variable_scope('Attention'):
                #att_in shape:[batchsize,maxseqlength,2*lstmunits]
                att_in =output
                size_per_head=16
                # QKV shape:[batchsize,maxseqlength,attentionsize]
                Q = self.attention_QKV(inputs=att_in,attentionSize=self.attentionSize,bias=False)
                Q = tf.reshape(Q, (-1, tf.shape(Q)[1], self.nhead, size_per_head))
                Q = tf.transpose(Q, [0, 2, 1, 3])

                K = self.attention_QKV(inputs=att_in,attentionSize=self.attentionSize,bias=False)
                K = tf.reshape(K, (-1, tf.shape(K)[1],self.nhead, size_per_head))
                K = tf.transpose(K, [0, 2, 1, 3])

                V = self.attention_QKV(inputs=att_in,attentionSize=self.attentionSize,bias=False)
                V = tf.reshape(V, (-1, tf.shape(V)[1], self.nhead, size_per_head))
                V = tf.transpose(V, [0, 2, 1, 3])

                # 计算内积，然后softmax
                A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
                A = tf.transpose(A, [0, 3, 2, 1])
                # A = Mask(A, V_len, mode='add')不需要mask，因为在load_batch的时候已经对太短或者太长的句子进行处理了
                A = tf.transpose(A, [0, 3, 2, 1])
                A = tf.nn.softmax(A)
                # 输出
                O = tf.matmul(A, V)
                O = tf.transpose(O, [0, 2, 1, 3])
                O = tf.reshape(O, (-1, tf.shape(O)[1], self.attentionSize))
                #O shape:[batchsize,maxseqlength,attentionsize]

            with tf.variable_scope('Prediction'):
                recall_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='recall_matrix'))
                pre_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='acc_matrix'))
                O=tf.reshape(O,(-1,self.maxSeqLength*self.attentionSize))

                #对O拉长为[batchsize,maxseqlength*attentionsize]
                #输出分类层
                weight = tf.Variable(
                    tf.random_normal([self.maxSeqLength*self.attentionSize, self.numClasses],
                                     stddev=0.1),
                    name='weight0'
                )
                bias = tf.Variable(tf.random_normal([self.numClasses], stddev=0.1), name='bias')
                y_ = (tf.matmul(O, weight) + bias)  # [batchsize,4]
                prediction = tf.argmax(y_, 1)
                correctPred = tf.equal(prediction, tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

                # 计算混淆矩阵
                confusion_matrix = tf.confusion_matrix(
                    labels=tf.argmax(labels, 1),  # 需要加一个 1
                    predictions=prediction,
                    num_classes=4,
                    dtype=tf.int32
                )
                # 横着除是召回率，纵着除是精确率
                col = tf.reduce_sum(confusion_matrix, 1)#每一行的和recall
                row = tf.reduce_sum(confusion_matrix, 0)#每一列的和pre

                recall_matrix = tf.add(recall_matrix, confusion_matrix / col)
                pre_matrix = tf.add(pre_matrix, confusion_matrix / row)

                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=labels)
                )

                tf.summary.scalar('loss', loss)
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('recall of \'-2\'', recall_matrix[0][0])
                tf.summary.scalar('recall of \'-1\'', recall_matrix[1][1])
                tf.summary.scalar('recall of \'-0\'', recall_matrix[2][2])
                tf.summary.scalar('recall of \'1\'', recall_matrix[3][3])
                tf.summary.scalar('precision of \'-2\'', pre_matrix[0][0])
                tf.summary.scalar('precision of \'-1\'', pre_matrix[1][1])
                tf.summary.scalar('precision of \'0\'', pre_matrix[2][2])
                tf.summary.scalar('precision of \'1\'', pre_matrix[3][3])
                tf.summary.scalar('F1 of \'-2\'', self.F1_score(pre_matrix[0][0],recall_matrix[0][0]))
                tf.summary.scalar('F1 of \'-1\'', self.F1_score(pre_matrix[1][1],recall_matrix[1][1]))
                tf.summary.scalar('F1 of \'0\'', self.F1_score(pre_matrix[2][2],recall_matrix[2][2]))
                tf.summary.scalar('F1 of \'1\'', self.F1_score(pre_matrix[3][3],recall_matrix[3][3]))
                merged = tf.summary.merge_all()

            with tf.variable_scope('Optimise'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)
        return graph

    def init_LSTM_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('Inputs'):
                labels = tf.placeholder(tf.float32, [None, self.numClasses], name='labels')
                input_data = tf.placeholder(tf.float32, [None, self.maxSeqLength, self.numDimensions],
                                            name='input_data')
                input_length = tf.placeholder(tf.int32, [None], name='input_length')

            with tf.variable_scope('LSTM'):
                encoder = tf.nn.rnn_cell.BasicLSTMCell(self.lstmUnits)
                encoder = tf.nn.rnn_cell.DropoutWrapper(cell=encoder,
                                                           output_keep_prob=self.dropOutRate)
                (
                    encoder_output,encoder_state
                ) = (
                    tf.nn.dynamic_rnn(cell=encoder,
                                      inputs=input_data,
                                      sequence_length=input_length,
                                      dtype=tf.float32
                                      )
                )
                #output shape：[batchsize,maxseqlength,lstmunits]
                output = encoder_output

            with tf.variable_scope('Prediction'):
                recall_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='recall_matrix'))
                pre_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='acc_matrix'))
                #对output拉长为[batchsize,maxseqlength*lstmunit]
                output=tf.reshape(output,(-1,self.maxSeqLength*self.lstmUnits))
                #输出分类层
                weight = tf.Variable(
                    tf.random_normal([self.maxSeqLength*self.lstmUnits, self.numClasses],
                                     stddev=0.1),
                    name='weight0'
                )
                bias = tf.Variable(tf.random_normal([self.numClasses], stddev=0.1), name='bias')
                y_ = (tf.matmul(output, weight) + bias)  # [batchsize,4]



                prediction = tf.argmax(y_, 1)
                correctPred = tf.equal(prediction, tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

                # 计算混淆矩阵
                confusion_matrix = tf.confusion_matrix(
                    labels=tf.argmax(labels, 1),  # 需要加一个 1
                    predictions=prediction,
                    num_classes=4,
                    dtype=tf.int32
                )
                # 横着除是召回率，纵着除是精确率
                col = tf.reduce_sum(confusion_matrix, 1)#每一行的和recall
                row = tf.reduce_sum(confusion_matrix, 0)#每一列的和pre

                recall_matrix = tf.add(recall_matrix, confusion_matrix / col)
                pre_matrix = tf.add(pre_matrix, confusion_matrix / row)

                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=labels)
                )
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('recall of \'-2\'', recall_matrix[0][0])
                tf.summary.scalar('recall of \'-1\'', recall_matrix[1][1])
                tf.summary.scalar('recall of \'-0\'', recall_matrix[2][2])
                tf.summary.scalar('recall of \'1\'', recall_matrix[3][3])
                tf.summary.scalar('precision of \'-2\'', pre_matrix[0][0])
                tf.summary.scalar('precision of \'-1\'', pre_matrix[1][1])
                tf.summary.scalar('precision of \'0\'', pre_matrix[2][2])
                tf.summary.scalar('precision of \'1\'', pre_matrix[3][3])
                tf.summary.scalar('F1 of \'-2\'', self.F1_score(pre_matrix[0][0],recall_matrix[0][0]))
                tf.summary.scalar('F1 of \'-1\'', self.F1_score(pre_matrix[1][1],recall_matrix[1][1]))
                tf.summary.scalar('F1 of \'0\'', self.F1_score(pre_matrix[2][2],recall_matrix[2][2]))
                tf.summary.scalar('F1 of \'1\'', self.F1_score(pre_matrix[3][3],recall_matrix[3][3]))
                merged = tf.summary.merge_all()

            with tf.variable_scope('Optimise'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)
        return graph

    def init_selfattention_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('Inputs'):
                labels = tf.placeholder(tf.float32, [None, self.numClasses], name='labels')
                input_data = tf.placeholder(tf.float32, [None, self.maxSeqLength, self.numDimensions],
                                            name='input_data')
                input_length = tf.placeholder(tf.int32, [None], name='input_length')

            with tf.variable_scope('Attention'):
                #att_in shape:[batchsize,maxseqlength,numDimensions]
                att_in =input_data
                size_per_head=16
                # QKV shape:[batchsize,maxseqlength,attentionsize]
                Q = self.attention_QKV(inputs=att_in,attentionSize=self.attentionSize,bias=False)
                Q = tf.reshape(Q, (-1, tf.shape(Q)[1], self.nhead, size_per_head))
                Q = tf.transpose(Q, [0, 2, 1, 3])

                K = self.attention_QKV(inputs=att_in,attentionSize=self.attentionSize,bias=False)
                K = tf.reshape(K, (-1, tf.shape(K)[1],self.nhead, size_per_head))
                K = tf.transpose(K, [0, 2, 1, 3])

                V = self.attention_QKV(inputs=att_in,attentionSize=self.attentionSize,bias=False)
                V = tf.reshape(V, (-1, tf.shape(V)[1], self.nhead, size_per_head))
                V = tf.transpose(V, [0, 2, 1, 3])

                # 计算内积，然后softmax
                A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
                A = tf.transpose(A, [0, 3, 2, 1])
                # A = Mask(A, V_len, mode='add')不需要mask，因为在load_batch的时候已经对太短或者太长的句子进行处理了
                A = tf.transpose(A, [0, 3, 2, 1])
                A = tf.nn.softmax(A)
                # 输出
                O = tf.matmul(A, V)
                O = tf.transpose(O, [0, 2, 1, 3])
                O = tf.reshape(O, (-1, tf.shape(O)[1], self.attentionSize))
                #O shape:[batchsize,maxseqlength,attentionsize]


            with tf.variable_scope('Prediction'):
                recall_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='recall_matrix'))
                pre_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='acc_matrix'))
                #对O拉长为[batchsize,maxseqlength*attentionsize]
                O=tf.reshape(O,(-1,self.maxSeqLength*self.attentionSize))
                #输出分类层
                weight = tf.Variable(
                    tf.random_normal([self.maxSeqLength*self.attentionSize, self.numClasses],
                                     stddev=0.1),
                    name='weight0'
                )
                bias = tf.Variable(tf.random_normal([self.numClasses], stddev=0.1), name='bias')
                y_ = (tf.matmul(O, weight) + bias)  # [batchsize,4]
                prediction = tf.argmax(y_, 1)
                correctPred = tf.equal(prediction, tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

                # 计算混淆矩阵
                confusion_matrix = tf.confusion_matrix(
                    labels=tf.argmax(labels, 1),  # 需要加一个 1
                    predictions=prediction,
                    num_classes=4,
                    dtype=tf.int32
                )
                # 横着除是召回率，纵着除是精确率
                col = tf.reduce_sum(confusion_matrix, 1)#每一行的和recall
                row = tf.reduce_sum(confusion_matrix, 0)#每一列的和pre

                recall_matrix = tf.add(recall_matrix, confusion_matrix / col)
                pre_matrix = tf.add(pre_matrix, confusion_matrix / row)

                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=labels)
                )

                tf.summary.scalar('loss', loss)
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('recall of \'-2\'', recall_matrix[0][0])
                tf.summary.scalar('recall of \'-1\'', recall_matrix[1][1])
                tf.summary.scalar('recall of \'-0\'', recall_matrix[2][2])
                tf.summary.scalar('recall of \'1\'', recall_matrix[3][3])
                tf.summary.scalar('precision of \'-2\'', pre_matrix[0][0])
                tf.summary.scalar('precision of \'-1\'', pre_matrix[1][1])
                tf.summary.scalar('precision of \'0\'', pre_matrix[2][2])
                tf.summary.scalar('precision of \'1\'', pre_matrix[3][3])
                tf.summary.scalar('F1 of \'-2\'', self.F1_score(pre_matrix[0][0],recall_matrix[0][0]))
                tf.summary.scalar('F1 of \'-1\'', self.F1_score(pre_matrix[1][1],recall_matrix[1][1]))
                tf.summary.scalar('F1 of \'0\'', self.F1_score(pre_matrix[2][2],recall_matrix[2][2]))
                tf.summary.scalar('F1 of \'1\'', self.F1_score(pre_matrix[3][3],recall_matrix[3][3]))
                merged = tf.summary.merge_all()

            with tf.variable_scope('Optimise'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)
        return graph
